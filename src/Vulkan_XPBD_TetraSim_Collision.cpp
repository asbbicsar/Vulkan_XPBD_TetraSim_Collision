#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const uint32_t TABLE_SIZE = 100003; // Hash table size (Prime number)
const uint32_t MAX_PER_CELL = 15;   // Maximum particles per cell

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};


// Ray Casting for Interaction
struct Ray {
    alignas(16) glm::vec3 origin;
    alignas(16) glm::vec3 direction;
};

struct InteractionState {
    int selectedNodeIndex;
    uint32_t minDist;
    float depth;
    float lambda;
};

struct MeshPushConstants {
    float dt;
    float u_Time;
    int subStepCnt;
    int iteration;
    int constraintOffset;
    int constraintCount;
	int isPressed;
    int pickPass;
    Ray mouseRay;
};


Ray getMouseRay(GLFWwindow* window, const glm::mat4& view, const glm::mat4& proj, float width, float height) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // 1. convert to NDC(Normalized Device Coordinates) (-1 to 1)
    float x = (2.0f * (float)xpos) / width - 1.0f;
    float y = (2.0f * (float)ypos) / height - 1.0f; // Vulkan/GLFW is Y-flipped

    // 2. calculate inverse of Projection * View matrix
    glm::mat4 invVP = glm::inverse(proj * view);

    // 3. convert points at Near and Far planes to World Coordinate
    glm::vec4 nearPos = invVP * glm::vec4(x, y, 0.0f, 1.0f);
    glm::vec4 farPos = invVP * glm::vec4(x, y, 1.0f, 1.0f);

    nearPos /= nearPos.w;
    farPos /= farPos.w;

    Ray ray;
    //ray.origin = glm::vec3(nearPos);
    ray.origin = glm::vec3(glm::inverse(view)[3]);
    ray.direction = glm::normalize(glm::vec3(farPos - nearPos));
    return ray;
};

struct PhysicsVertex {
    glm::vec4 pos;

    glm::vec4 vel;
    glm::vec4 prevPos;
    glm::vec4 normal;

    float invMass;
    float isFixed;
    float stress;
    float padding;

    glm::vec4 colData; //(x: radius, y : objectID, z : padding, w : padding)
};

struct RenderVertex {
    glm::vec4 color;
};

struct Edge {
    uint32_t indices[2];
    int32_t marker;
    int32_t padding;
};

struct Tetrahedron {
    uint32_t indices[4];
};

struct Face {
    uint32_t indices[3];
    int32_t marker;
};

struct DistanceConstraint {
    uint32_t p1, p2;
    float restLen;
    float compliance;
    float lambda;
    float padding[3];
};

struct VolumeConstraint {
    uint32_t p1, p2, p3, p4;
    float restVol;
    float compliance;
    float lambda;
    float padding;
};

bool getValidLine(std::ifstream& file, std::string& line) {
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        if (line[first] == '#') continue;

        return true;
    }

    return false;
}

bool parseNodeFile(const std::string& filename, std::vector<PhysicsVertex>& outPhys, std::vector<RenderVertex>& outRends, int& outStartIndex) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open NODE file: " << filename << std::endl;
        return false;
    }

    std::string line;
    if (!getValidLine(file, line)) return false;

    // 헤더 파싱: [점의 개수] [차원] [속성 개수] [경계 마커 개수]
    int numPoints, dim, numAttributes, numBoundaryMarkers;
    std::istringstream headerSS(line);
    headerSS >> numPoints >> dim >> numAttributes >> numBoundaryMarkers;

    outPhys.reserve(numPoints);
    outRends.reserve(numPoints);
    bool isFirstNode = true;

    for (int i = 0; i < numPoints; ++i) {
        if (!getValidLine(file, line)) break;
        std::istringstream iss(line);

        int index;
        PhysicsVertex p;
        iss >> index >> p.pos.x >> p.pos.y >> p.pos.z;
        p.vel = glm::vec4(0.0f);
        p.prevPos = p.pos;

        p.invMass = float(numPoints); // 기본 역질량 설정
        p.isFixed = 0.0f;
		p.stress = 0.0f;

        RenderVertex r;
		r.color = glm::vec4(p.pos.x, p.pos.y, p.pos.z, 1.0f);

        // 첫 번째 정점의 인덱스가 0인지 1인지 기록 (ele 파일 파싱 시 사용)
        if (isFirstNode) {
            outStartIndex = index;
            isFirstNode = false;
        }

        outPhys.push_back(p);
		outRends.push_back(r);
    }

    std::cout << "Loaded " << outPhys.size() << " physics vertices." << std::endl;
    return true;
}

// .ele 파일 파싱
bool parseEleFile(const std::string& filename, int startIndex, std::vector<Tetrahedron>& outTetras) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open ELE file: " << filename << std::endl;
        return false;
    }

    std::string line;
    if (!getValidLine(file, line)) return false;

    // 헤더 파싱: [사면체 개수] [정점 개수(보통 4)] [속성 개수]
    int numTetras, nodesPerTetra, numAttributes;
    std::istringstream headerSS(line);
    headerSS >> numTetras >> nodesPerTetra >> numAttributes;

    if (nodesPerTetra != 4) {
        std::cerr << "Error: Not a tetrahedral mesh (nodes per element != 4)" << std::endl;
        return false;
    }

    outTetras.reserve(numTetras);

    for (int i = 0; i < numTetras; ++i) {
        if (!getValidLine(file, line)) break;
        std::istringstream ss(line);

        int index;
        Tetrahedron tet;

        ss >> index >> tet.indices[0] >> tet.indices[1] >> tet.indices[2] >> tet.indices[3];

        // 0-based 인덱싱으로 정규화 (Vulkan 배열 접근을 위해)
        for (int j = 0; j < 4; ++j) {
            tet.indices[j] -= startIndex;
        }

        outTetras.push_back(tet);
    }

    std::cout << "Loaded " << outTetras.size() << " tetrahedra." << std::endl;
    return true;
}

// .edge 파일 파싱
bool parseEdgeFile(const std::string& filename, int startIndex, std::vector<Edge>& outEdges) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open EDGE file: " << filename << std::endl;
        return false;
    }

    std::string line;
    if (!getValidLine(file, line)) return false;

    // 헤더 파싱: [사면체 개수] [정점 개수(보통 4)] [속성 개수]
    int numEdges, boundaryMarker;
    std::istringstream headerSS(line);
    headerSS >> numEdges >> boundaryMarker;

    outEdges.reserve(numEdges);

    for (int i = 0; i < numEdges; ++i) {
        if (!getValidLine(file, line)) break;
        std::istringstream ss(line);

        int index;
        Edge edge;

        ss >> index >> edge.indices[0] >> edge.indices[1] >> edge.marker;

        // 0-based 인덱싱으로 정규화 (Vulkan 배열 접근을 위해)
        for (int j = 0; j < 2; ++j) {
            edge.indices[j] -= startIndex;
        }

        outEdges.push_back(edge);
    }

    std::cout << "Loaded " << outEdges.size() << " edges." << std::endl;
    return true;
}

bool parseFaceFile(const std::string& filename, int startIndex, std::vector<Face>& outFaces) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open FACE file: " << filename << std::endl;
        return false;
    }

    std::string line;
    if (!getValidLine(file, line)) return false;

    // 헤더 파싱: [사면체 개수] [정점 개수(보통 4)] [속성 개수]
    int numEdges, boundaryMarker;
    std::istringstream headerSS(line);
    headerSS >> numEdges >> boundaryMarker;

    outFaces.reserve(numEdges);

    for (int i = 0; i < numEdges; ++i) {
        if (!getValidLine(file, line)) break;
        std::istringstream ss(line);

        int index;
        Face face;

        ss >> index >> face.indices[0] >> face.indices[1] >> face.indices[2] >> face.marker;

        // 0-based 인덱싱으로 정규화 (Vulkan 배열 접근을 위해)
        for (int j = 0; j < 3; ++j) {
            face.indices[j] -= startIndex;
        }

        outFaces.push_back(face);
    }

    std::cout << "Loaded " << outFaces.size() << " faces." << std::endl;
    return true;
}

bool generateDistanceConstraints(const std::vector<PhysicsVertex>& phys, const std::vector<Edge>& edges, const float& stiffness, std::vector<DistanceConstraint>& outConstraints) {
    for (const auto& edge : edges) {
        DistanceConstraint dc{};
        dc.p1 = edge.indices[0];
        dc.p2 = edge.indices[1];
        dc.restLen = glm::length(glm::vec3(phys[dc.p1].pos - phys[dc.p2].pos));
        dc.compliance = dc.restLen / stiffness;
        dc.lambda = 0.0f;
        outConstraints.push_back(dc);
    }
    return true;
}

bool generateVolumeConstraints(const std::vector<PhysicsVertex>& phys, const std::vector<Tetrahedron>& tetras, const float& stiffness, std::vector<VolumeConstraint>& outConstraints) {
    for (const auto& tetra : tetras) {
        VolumeConstraint vc{};
        vc.p1 = tetra.indices[0];
        vc.p2 = tetra.indices[1];
        vc.p3 = tetra.indices[2];
        vc.p4 = tetra.indices[3];
        vc.restVol = glm::dot(glm::cross(glm::vec3(phys[vc.p2].pos - phys[vc.p1].pos),
            glm::vec3(phys[vc.p3].pos - phys[vc.p1].pos)),
            glm::vec3(phys[vc.p4].pos - phys[vc.p1].pos)) / 6.0f;
        vc.compliance = vc.restVol / stiffness;
        vc.lambda = 0.0f;
        outConstraints.push_back(vc);
    }
    return true;
}

struct ColorGroup {
    uint32_t offset;
    uint32_t count;
};

struct ColoredDistanceResult {
    std::vector<DistanceConstraint> reorderedConstraints;
    std::vector<ColorGroup> groups;
};

struct ColoredVolumeResult {
    std::vector<VolumeConstraint> reorderedConstraints;
    std::vector<ColorGroup> groups;
};

// 1. 거리 제약 조건 컬러링
ColoredDistanceResult colorDistanceConstraints(const std::vector<DistanceConstraint>& constraints, uint32_t numParticles) {
    std::vector<std::vector<DistanceConstraint>> colorGroups;
    std::vector<std::vector<bool>> particleUsed;

    for (const auto& c : constraints) {
        int color = 0;
        while (true) {
            // 새로운 색상이 필요하면 추가
            if (color == colorGroups.size()) {
                colorGroups.push_back(std::vector<DistanceConstraint>());
                particleUsed.push_back(std::vector<bool>(numParticles, false));
            }

            // 현재 색상 그룹에서 해당 정점들이 이미 사용 중인지 확인
            if (!particleUsed[color][c.p1] && !particleUsed[color][c.p2]) {
                colorGroups[color].push_back(c);
                particleUsed[color][c.p1] = true;
                particleUsed[color][c.p2] = true;
                break; // 색상을 찾았으므로 다음 제약 조건으로 넘어감
            }
            color++;
        }
    }

    ColoredDistanceResult result;
    uint32_t currentOffset = 0;
    for (const auto& group : colorGroups) {
        ColorGroup cg;
        cg.offset = currentOffset;
        cg.count = static_cast<uint32_t>(group.size());
        result.groups.push_back(cg);

        result.reorderedConstraints.insert(result.reorderedConstraints.end(), group.begin(), group.end());
        currentOffset += cg.count;
    }
    return result;
}

// 2. 부피 제약 조건 컬러링
ColoredVolumeResult colorVolumeConstraints(const std::vector<VolumeConstraint>& constraints, uint32_t numParticles) {
    std::vector<std::vector<VolumeConstraint>> colorGroups;
    std::vector<std::vector<bool>> particleUsed;

    for (const auto& c : constraints) {
        int color = 0;
        while (true) {
            if (color == colorGroups.size()) {
                colorGroups.push_back(std::vector<VolumeConstraint>());
                particleUsed.push_back(std::vector<bool>(numParticles, false));
            }

            if (!particleUsed[color][c.p1] && !particleUsed[color][c.p2] &&
                !particleUsed[color][c.p3] && !particleUsed[color][c.p4]) {
                colorGroups[color].push_back(c);
                particleUsed[color][c.p1] = true;
                particleUsed[color][c.p2] = true;
                particleUsed[color][c.p3] = true;
                particleUsed[color][c.p4] = true;
                break;
            }
            color++;
        }
    }

    ColoredVolumeResult result;
    uint32_t currentOffset = 0;
    for (const auto& group : colorGroups) {
        ColorGroup cg;
        cg.offset = currentOffset;
        cg.count = static_cast<uint32_t>(group.size());
        result.groups.push_back(cg);

        result.reorderedConstraints.insert(result.reorderedConstraints.end(), group.begin(), group.end());
        currentOffset += cg.count;
    }
    return result;
}

struct HashCell {
    uint32_t count;
    uint32_t pIndices[MAX_PER_CELL];
};


class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    //depthImage, depthImageMemory;
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    VkRenderPass renderPass;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkPipelineLayout computePipelineLayout;
    VkPipeline pickPipeline;
    VkPipeline predictPipeline;
    VkPipeline hashClearPipeline;
    VkPipeline hashInsertPipeline;
    VkPipeline solveMousePipeline;
    VkPipeline solveDistPipeline;
    VkPipeline solveVolPipeline;
    VkPipeline solveCollisionPipeline;
    VkPipeline updatePipeline;

    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;

    std::vector<PhysicsVertex> physicsVerts;
	std::vector<RenderVertex> renderVerts;
    std::vector<Tetrahedron> tetras;
    std::vector<Edge> edges;
    std::vector<Face> faces;
    std::vector<HashCell> cells;

    std::vector<uint32_t> indices;

    std::vector<VolumeConstraint> volumeConstraints;
    std::vector<DistanceConstraint> distanceConstraints;

    std::vector<ColorGroup> distanceColorGroups;
    std::vector<ColorGroup> volumeColorGroups;

    std::vector<VkBuffer> physicsVertexBuffers;
    std::vector<VkDeviceMemory> physicsVertexBuffersMemory;
    std::vector<VkBuffer> renderVertexBuffers;
    std::vector<VkDeviceMemory> renderVertexBuffersMemory;
    std::vector<VkBuffer> interactionBuffers;
    std::vector<VkDeviceMemory> interactionBuffersMemory;
    std::vector<VkBuffer> hashBuffers;
    std::vector<VkDeviceMemory> hashBuffersMemory;
    std::vector<void*> interactionBuffersMapped;

    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkBuffer tetraBuffer;
    VkDeviceMemory tetraBufferMemory;
    VkBuffer edgeBuffer;
    VkDeviceMemory edgeBufferMemory;
    VkBuffer faceBuffer;
    VkDeviceMemory faceBufferMemory;
    VkBuffer volumeConstraintsBuffer;
    VkDeviceMemory volumeConstraintsBufferMemory;
    VkBuffer distanceConstraintsBuffer;
    VkDeviceMemory distanceConstraintsBufferMemory;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> computeDescriptorSets;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;

    UniformBufferObject ubo{};

    const float stiffness = 512.0f;

    bool framebufferResized = false;

    void loadMesh(const std::string& nodeFile, 
        const std::string& eleFile, 
        const std::string& edgeFile, 
        const std::string& faceFile,
        glm::vec3 initialOffset = glm::vec3(0.0f),
        float objectID = 0.0f) 
    {
        int startIndex;
        uint32_t indexOffset = physicsVerts.size(); // 기존 파티클 개수를 오프셋으로 사용

        std::vector<PhysicsVertex> tempPhysicsVerts;
		std::vector<RenderVertex> tempRenderVerts;
        std::vector<Tetrahedron> tempTetras;
        std::vector<Edge> tempEdges;
        std::vector<Face> tempFaces;

        if (!parseNodeFile(nodeFile, tempPhysicsVerts, tempRenderVerts, startIndex)) {
            throw std::runtime_error("Failed to load node file!");
        }

        // 초기 위치 조정 (두 객체가 겹치지 않게 하기 위함)
        for (auto& p : tempPhysicsVerts) {
            p.pos.x += initialOffset.x;
            p.pos.y += initialOffset.y;
            p.pos.z += initialOffset.z;
            p.prevPos = p.pos;

            p.colData = glm::vec4(0.05f, objectID, 0.0f, 0.0f);
        }
        physicsVerts.insert(physicsVerts.end(), tempPhysicsVerts.begin(), tempPhysicsVerts.end());
        renderVerts.insert(renderVerts.end(), tempRenderVerts.begin(), tempRenderVerts.end());
        if (!parseEleFile(eleFile, startIndex, tempTetras)) { /* 에러 처리 */ }
        for (auto& tet : tempTetras) {
            for (int i = 0; i < 4; ++i) tet.indices[i] += indexOffset;
            tetras.push_back(tet);
        }

        if (!parseEdgeFile(edgeFile, startIndex, tempEdges)) { /* 에러 처리 */ }
        for (auto& edge : tempEdges) {
            for (int i = 0; i < 2; ++i) edge.indices[i] += indexOffset;
            edges.push_back(edge);
        }

        if (!parseFaceFile(faceFile, startIndex, tempFaces)) { /* 에러 처리 */ }
        for (auto& face : tempFaces) {
            for (int i = 0; i < 3; ++i) {
                face.indices[i] += indexOffset;
                indices.push_back(face.indices[i]); // 렌더링용 index buffer
            }
            faces.push_back(face);
        }
    }

    void initHashCells()
    {
        cells.resize(TABLE_SIZE);
        for (auto& cell : cells) {
            cell.count = 0;
        }
    }
    
    void createConstraints() {
        if (!generateDistanceConstraints(physicsVerts, edges, stiffness, distanceConstraints)) {
            throw std::runtime_error("Failed to generate distance constraints!");
        }
        if (!generateVolumeConstraints(physicsVerts, tetras, stiffness, volumeConstraints)) {
            throw std::runtime_error("Failed to generate volume constraints!");
        }

        auto distResult = colorDistanceConstraints(distanceConstraints, physicsVerts.size());
        distanceConstraints = distResult.reorderedConstraints;
        distanceColorGroups = distResult.groups;

        auto volResult = colorVolumeConstraints(volumeConstraints, physicsVerts.size());
        volumeConstraints = volResult.reorderedConstraints;
        volumeColorGroups = volResult.groups;

        std::cout << "Distance Colors: " << distanceColorGroups.size()
            << ", Volume Colors: " << volumeColorGroups.size() << std::endl;
    }

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "XPBD TetraSim Collision", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan() {
        loadMesh(
            "models/bunny_1k.1.node",
            "models/bunny_1k.1.ele",
            "models/bunny_1k.1.edge",
            "models/bunny_1k.1.face",
            glm::vec3(0.0f, 3.0f, 0.0f),
            1.0f);
        loadMesh(
            "models/bunny_1k.1.node",
            "models/bunny_1k.1.ele",
            "models/bunny_1k.1.edge",
            "models/bunny_1k.1.face",
            glm::vec3(0.0f, 0.0f, 0.0f),
            2.0f);
        initHashCells();
        createConstraints();
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createDepthResources();
        createRenderPass();
        createComputeDescriptorSetLayout();
        createRenderDescriptorSetLayout();
        //createComputePipeline();
        createComputePipelines();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createShaderStorageBuffers();
        createInteractionBuffer();
		createHashBuffer();
        createDistanceConstraintBuffer();
        createVolumeConstraintBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createComputeDescriptorSets();
        createRenderDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    void cleanupSwapChain() {
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void cleanup() {
        cleanupSwapChain();

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyPipeline(device, pickPipeline, nullptr);
        vkDestroyPipeline(device, predictPipeline, nullptr);
        vkDestroyPipeline(device, hashClearPipeline, nullptr);
        vkDestroyPipeline(device, hashInsertPipeline, nullptr);
        vkDestroyPipeline(device, solveMousePipeline, nullptr);
        vkDestroyPipeline(device, solveDistPipeline, nullptr);
        vkDestroyPipeline(device, solveVolPipeline, nullptr);
        vkDestroyPipeline(device, solveCollisionPipeline, nullptr);
        vkDestroyPipeline(device, updatePipeline, nullptr);
        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, physicsVertexBuffers[i], nullptr);
            vkFreeMemory(device, physicsVertexBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, renderVertexBuffers[i], nullptr);
            vkFreeMemory(device, renderVertexBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, interactionBuffers[i], nullptr);
            vkFreeMemory(device, interactionBuffersMemory[i], nullptr);
            vkDestroyBuffer(device, hashBuffers[i], nullptr);
            vkFreeMemory(device, hashBuffersMemory[i], nullptr);
        }

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);
        vkDestroyBuffer(device, distanceConstraintsBuffer, nullptr);
        vkFreeMemory(device, distanceConstraintsBufferMemory, nullptr);
        vkDestroyBuffer(device, volumeConstraintsBuffer, nullptr);
        vkFreeMemory(device, volumeConstraintsBufferMemory, nullptr);

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        for (auto semaphore : renderFinishedSemaphores) {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    void createDepthResources() {
        VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

        // 1. 이미지 생성
        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat,
            VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);

        // 2. 이미지 뷰 생성
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = depthImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = depthFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &viewInfo, nullptr, &depthImageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create depth image view!");
        }
    }

    // 헬퍼 함수 (기존 createBuffer와 유사하게 이미지 생성용으로 구현 필요)
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
        VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
        VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCreateImage(device, &imageInfo, nullptr, &image);

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory);
        vkBindImageMemory(device, image, imageMemory, 0);
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = VK_FORMAT_D32_SFLOAT; // 선택한 포맷
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;  // 매 프레임 초기화
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1; // 0번은 Color, 1번은 Depth
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }
    void createRenderDescriptorSetLayout() {
        std::array<VkDescriptorSetLayoutBinding, 3> layoutBindings{};
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        layoutBindings[2].binding = 2;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[2].pImmutableSamplers = nullptr;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render descriptor set layout!");
        }
    }

    void createComputePipelines() {
        // 1. 파이프라인 레이아웃 생성 (3개 파이프라인이 공유)
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(MeshPushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout);

        // 2. 개별 파이프라인 생성 헬퍼 함수
        auto createPipeline = [&](const std::string& shaderPath, VkPipeline& pipeline) {
            auto shaderCode = readFile(shaderPath);
            VkShaderModule shaderModule = createShaderModule(shaderCode);

            VkPipelineShaderStageCreateInfo stageInfo{};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = shaderModule;
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = computePipelineLayout;
            pipelineInfo.stage = stageInfo;

            if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create compute pipeline: " + shaderPath);
            }
            vkDestroyShaderModule(device, shaderModule, nullptr);
            };

        // 3. 실제 파이프라인 생성 호출
        createPipeline("shaders/Pick.comp.spv", pickPipeline);
        createPipeline("shaders/Predict.comp.spv", predictPipeline);
		createPipeline("shaders/HashClear.comp.spv", hashClearPipeline);
		createPipeline("shaders/HashInsert.comp.spv", hashInsertPipeline);
        createPipeline("shaders/SolveMouse.comp.spv", solveMousePipeline);
        createPipeline("shaders/SolveDist.comp.spv", solveDistPipeline);
        createPipeline("shaders/SolveVol.comp.spv", solveVolPipeline);
        createPipeline("shaders/SolveCollision.comp.spv", solveCollisionPipeline);
        createPipeline("shaders/Update.comp.spv", updatePipeline);
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("shaders/VertexShader.vert.spv");
        auto fragShaderCode = readFile("shaders/FragmentShader.frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS; // 앞의 것만 통과
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<VkImageView, 2> attachments = {
                swapChainImageViews[i],
                depthImageView
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics command pool!");
        }
    }

    void createShaderStorageBuffers() {
        createPhysicsVertexBuffer();
        createRenderVertexBuffer();
    }

    void createPhysicsVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(PhysicsVertex) * physicsVerts.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory
        );

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, physicsVerts.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        physicsVertexBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        physicsVertexBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(
                bufferSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                physicsVertexBuffers[i],
                physicsVertexBuffersMemory[i]
            );
            copyBuffer(stagingBuffer, physicsVertexBuffers[i], bufferSize);
        }
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createRenderVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(RenderVertex) * renderVerts.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory
        );

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, renderVerts.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        renderVertexBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        renderVertexBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(
                bufferSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                renderVertexBuffers[i],
                renderVertexBuffersMemory[i]
            );
            copyBuffer(stagingBuffer, renderVertexBuffers[i], bufferSize);
        }
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createInteractionBuffer() {
        VkDeviceSize bufferSize = sizeof(InteractionState);

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory
        );

        InteractionState initialState;
        initialState.selectedNodeIndex = -1;
        initialState.minDist = 0xFFFFFFFF;

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, &initialState, (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        interactionBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        interactionBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(
                bufferSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                interactionBuffers[i],
                interactionBuffersMemory[i]
            );
            copyBuffer(stagingBuffer, interactionBuffers[i], bufferSize);
        }
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createHashBuffer() {
        VkDeviceSize bufferSize = sizeof(HashCell) * TABLE_SIZE;

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory
        );

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, cells.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        hashBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        hashBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(
                bufferSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                hashBuffers[i],
                hashBuffersMemory[i]
            );
            copyBuffer(stagingBuffer, hashBuffers[i], bufferSize);
        }
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }


    void createDistanceConstraintBuffer() {
        VkDeviceSize CONSTRAINT_COUNT = distanceConstraints.size();
        std::vector<DistanceConstraint> constraints(CONSTRAINT_COUNT);

        VkDeviceSize bufferSize = sizeof(DistanceConstraint) * constraints.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory
        );

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, distanceConstraints.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            distanceConstraintsBuffer,
            distanceConstraintsBufferMemory
        );
        copyBuffer(stagingBuffer, distanceConstraintsBuffer, bufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createVolumeConstraintBuffer() {
        VkDeviceSize CONSTRAINT_COUNT = volumeConstraints.size();
        std::vector<VolumeConstraint> constraints(CONSTRAINT_COUNT);

        VkDeviceSize bufferSize = sizeof(VolumeConstraint) * constraints.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory
        );

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, volumeConstraints.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);


        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            volumeConstraintsBuffer,
            volumeConstraintsBufferMemory
        );
        copyBuffer(stagingBuffer, volumeConstraintsBuffer, bufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(uint32_t) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 6;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();

        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createRenderDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorBufferInfo physicsInfo{};
            physicsInfo.buffer = physicsVertexBuffers[i];
            physicsInfo.offset = 0;
            physicsInfo.range = sizeof(PhysicsVertex) * physicsVerts.size();

            VkDescriptorBufferInfo renderInfo{};
            renderInfo.buffer = renderVertexBuffers[i];
            renderInfo.offset = 0;
            renderInfo.range = sizeof(RenderVertex) * renderVerts.size();

            std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &physicsInfo;

            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = descriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &renderInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }
    void createComputeDescriptorSetLayout() {
        std::array<VkDescriptorSetLayoutBinding, 6> layoutBindings{};
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[2].binding = 2;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[3].binding = 3;
        layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[3].descriptorCount = 1;
        layoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[4].binding = 4;
        layoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[4].descriptorCount = 1;
        layoutBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[5].binding = 5;
        layoutBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[5].descriptorCount = 1;
        layoutBindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 6;
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }

    void createComputeDescriptorSets() {
        computeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        computeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, computeDescriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate compute descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo inBufferInfo{};
            inBufferInfo.buffer = physicsVertexBuffers[(i + 1) % MAX_FRAMES_IN_FLIGHT]; // 우리가 만든 SSBO
            inBufferInfo.offset = 0;
            inBufferInfo.range = sizeof(PhysicsVertex) * physicsVerts.size();

            VkDescriptorBufferInfo outBufferInfo{};
            outBufferInfo.buffer = physicsVertexBuffers[i]; // 우리가 만든 SSBO
            outBufferInfo.offset = 0;
            outBufferInfo.range = sizeof(PhysicsVertex) * physicsVerts.size();

            VkDescriptorBufferInfo DistanceConstraintBufferInfo{};
            DistanceConstraintBufferInfo.buffer = distanceConstraintsBuffer;
            DistanceConstraintBufferInfo.offset = 0;
            DistanceConstraintBufferInfo.range = sizeof(DistanceConstraint) * distanceConstraints.size();

            VkDescriptorBufferInfo VolumeConstraintBufferInfo{};
            VolumeConstraintBufferInfo.buffer = volumeConstraintsBuffer;
            VolumeConstraintBufferInfo.offset = 0;
            VolumeConstraintBufferInfo.range = sizeof(VolumeConstraint) * volumeConstraints.size();

			VkDescriptorBufferInfo InteractionBufferInfo{};
			InteractionBufferInfo.buffer = interactionBuffers[i];
            InteractionBufferInfo.offset = 0;
			InteractionBufferInfo.range = sizeof(InteractionState);

            VkDescriptorBufferInfo HashBufferInfo{};
            HashBufferInfo.buffer = hashBuffers[i];
            HashBufferInfo.offset = 0;
            HashBufferInfo.range = sizeof(HashCell) * TABLE_SIZE;

            std::array<VkWriteDescriptorSet, 6> descriptorWrites{};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = computeDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &inBufferInfo;
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = computeDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &outBufferInfo;
            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = computeDescriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &DistanceConstraintBufferInfo;
            descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[3].dstSet = computeDescriptorSets[i];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].pBufferInfo = &VolumeConstraintBufferInfo;
            descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[4].dstSet = computeDescriptorSets[i];
            descriptorWrites[4].dstBinding = 4;
            descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].pBufferInfo = &InteractionBufferInfo;
			descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[5].dstSet = computeDescriptorSets[i];
			descriptorWrites[5].dstBinding = 5;
			descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[5].descriptorCount = 1;
            descriptorWrites[5].pBufferInfo = &HashBufferInfo;

            vkUpdateDescriptorSets(device, 6, descriptorWrites.data(), 0, nullptr);
        }
    }
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, MeshPushConstants& pc) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        pc.subStepCnt = 16;

        uint32_t readIdx = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        uint32_t writeIdx = currentFrame;

        static int pickSyncCounter = 0; // 동기화를 위한 카운터 변수 추가

        if (pc.pickPass == 1) { // trigger pick
            vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pickPipeline);
            vkCmdBindDescriptorSets(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[writeIdx], 0, nullptr);

            // Pass 0: reset interaction buffer
            pc.pickPass = 0;
            vkCmdPushConstants(commandBuffers[currentFrame], computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshPushConstants), &pc);
            vkCmdDispatch(commandBuffers[currentFrame], 1, 1, 1);
            addComputeBarrier(commandBuffers[currentFrame], interactionBuffers[writeIdx]);

			// Pass 1: find out the closest node
            pc.pickPass = 1;
            vkCmdPushConstants(commandBuffers[currentFrame], computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshPushConstants), &pc);
            vkCmdDispatch(commandBuffers[currentFrame], (physicsVerts.size() + 63) / 64, 1, 1);
            addComputeBarrier(commandBuffers[currentFrame], interactionBuffers[writeIdx]);

            // Pass 2: confirm the picked node index
            pc.pickPass = 2;
            vkCmdPushConstants(commandBuffers[currentFrame], computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshPushConstants), &pc);
            vkCmdDispatch(commandBuffers[currentFrame], (physicsVerts.size() + 63) / 64, 1, 1);
            addComputeBarrier(commandBuffers[currentFrame], interactionBuffers[writeIdx]);

            VkBufferCopy copyRegion{};
            copyRegion.size = sizeof(InteractionState);
            vkCmdCopyBuffer(commandBuffer, interactionBuffers[writeIdx], interactionBuffers[readIdx], 1, &copyRegion);

            VkBufferMemoryBarrier interactionBarrier{};
            interactionBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            interactionBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            interactionBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            interactionBarrier.buffer = interactionBuffers[readIdx];
            interactionBarrier.offset = 0;
            interactionBarrier.size = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &interactionBarrier, 0, nullptr);
        }
        else if (pc.pickPass == 0) { // mouse released
            vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pickPipeline);
            vkCmdBindDescriptorSets(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[writeIdx], 0, nullptr);

            // Pass 0: reset interaction buffer
            pc.pickPass = 0;
            vkCmdPushConstants(commandBuffers[currentFrame], computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshPushConstants), &pc);
            vkCmdDispatch(commandBuffers[currentFrame], 1, 1, 1);
            addComputeBarrier(commandBuffers[currentFrame], interactionBuffers[writeIdx]);

            VkBufferCopy copyRegion{};
            copyRegion.size = sizeof(InteractionState);
            vkCmdCopyBuffer(commandBuffer, interactionBuffers[writeIdx], interactionBuffers[readIdx], 1, &copyRegion);

            VkBufferMemoryBarrier interactionBarrier{};
            interactionBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            interactionBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            interactionBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            interactionBarrier.buffer = interactionBuffers[readIdx];
            interactionBarrier.offset = 0;
            interactionBarrier.size = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &interactionBarrier, 0, nullptr);
        }

        for (int i = 0; i < pc.subStepCnt; i++) {

            // Step 1. Predict
            pc.iteration = 0;
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, predictPipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout,
                0, 1, &computeDescriptorSets[writeIdx], 0, nullptr);
            vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshPushConstants), &pc);
            vkCmdDispatch(commandBuffer, (physicsVerts.size() + 63) / 64, 1, 1);
            addComputeBarrier(commandBuffer, physicsVertexBuffers[writeIdx]);

			// Step 1.5. Build spatial hash
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, hashClearPipeline);
            vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshPushConstants), &pc);
            vkCmdDispatch(commandBuffer, (TABLE_SIZE + 63) / 64, 1, 1);
            addComputeBarrier(commandBuffer, hashBuffers[writeIdx]);
            
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, hashInsertPipeline);
            vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshPushConstants), &pc);
            vkCmdDispatch(commandBuffer, (physicsVerts.size() + 63) / 64, 1, 1);
            addComputeBarrier(commandBuffer, hashBuffers[writeIdx]);

            // Step 2. Solve (Iterative)
            int constraintCount = 4;
            for (int j = 0; j < constraintCount; j++) {
                pc.iteration = j;

                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, solveMousePipeline);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[writeIdx], 0, nullptr);
                vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshPushConstants), &pc);
                vkCmdDispatch(commandBuffer, 1, 1, 1);
                addComputeBarrier(commandBuffer, physicsVertexBuffers[writeIdx]);

                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, solveDistPipeline);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout,
                    0, 1, &computeDescriptorSets[writeIdx], 0, nullptr);

                for (const auto& group : distanceColorGroups) {
                    pc.constraintOffset = group.offset;
                    pc.constraintCount = group.count;
                    vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshPushConstants), &pc);
                    vkCmdDispatch(commandBuffer, (group.count + 63) / 64, 1, 1);
                    addComputeBarrier(commandBuffer, physicsVertexBuffers[writeIdx]);
                }

                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, solveVolPipeline);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout,
                    0, 1, &computeDescriptorSets[writeIdx], 0, nullptr);
                for (const auto& group : volumeColorGroups) {
                    pc.constraintOffset = group.offset;
                    pc.constraintCount = group.count;
                    vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshPushConstants), &pc);
                    vkCmdDispatch(commandBuffer, (group.count + 63) / 64, 1, 1);
                    addComputeBarrier(commandBuffer, physicsVertexBuffers[writeIdx]);
                }

                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, solveCollisionPipeline);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[writeIdx], 0, nullptr);
                vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshPushConstants), &pc);
                vkCmdDispatch(commandBuffer, (physicsVerts.size() + 63) / 64, 1, 1);
                addComputeBarrier(commandBuffer, physicsVertexBuffers[writeIdx]);
            }
            // Step 3. Update
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, updatePipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout,
                0, 1, &computeDescriptorSets[readIdx], 0, nullptr); // Update는 readIdx를 읽고 writeIdx에 쓰는 형태이므로, readIdx로 바인딩
            vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeshPushConstants), &pc);
            vkCmdDispatch(commandBuffer, (physicsVerts.size() + 63) / 64, 1, 1);
            addComputeBarrier(commandBuffer, physicsVertexBuffers[readIdx]);
        }


        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = sizeof(PhysicsVertex) * physicsVerts.size();
        
        vkCmdCopyBuffer(
            commandBuffer,
            physicsVertexBuffers[readIdx],
            physicsVertexBuffers[writeIdx],
            1,
            &copyRegion
        );
        
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = physicsVertexBuffers[writeIdx];
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;
        
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &barrier, 0, nullptr
        );

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearValues[1].depthStencil = { 1.0f, 0 }; // 깊이 최대값(1.0)으로 초기화

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);


        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        vkCmdEndRenderPass(commandBuffer);
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void addComputeBarrier(VkCommandBuffer commandBuffer, VkBuffer buffer) {
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;       // Compute에서 씀
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; // Vertex에서 읽음
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = buffer;      // 우리가 만든 SSBO
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // 출발지: Compute 단계
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,    // 목적지: Vertex 입력 단계
            0, 0, nullptr, 1, &barrier, 0, nullptr
        );
    }

    void createSyncObjects() {
        uint32_t imageCount = static_cast<uint32_t>(swapChainImages.size());

        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(imageCount);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        // Acquire용 세마포어와 Fence 생성
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]);
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]);
        }

        // RenderFinished용 세마포어 생성 (이미지당 1개)
        for (size_t i = 0; i < imageCount; i++) {
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]);
        }
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        ubo.model = glm::mat4(1.0f);
        ubo.view = glm::lookAt(glm::vec3(4.0f, 1.0f, -4.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void drawFrame() {
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        updateUniformBuffer(currentFrame);

        MeshPushConstants pc{};
        pc.dt = 0.016f;
        static float time = 0.0f;
        time += pc.dt;
        pc.u_Time = time;

        static bool wasPressed = false;
        bool isPressedNow = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);

        bool triggerPick = (isPressedNow && !wasPressed);
        bool isReleased = (!isPressedNow && wasPressed);
        wasPressed = isPressedNow;

        pc.isPressed = isPressedNow ? 1 : 0;

        if (isPressedNow) {
            pc.mouseRay = getMouseRay(window, ubo.view, ubo.proj, WIDTH, HEIGHT);
        }

        if (triggerPick)
            pc.pickPass = 1;
        else if (isReleased)
            pc.pickPass = 0;
        else
            pc.pickPass = 2;


        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex, pc);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[imageIndex] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;

        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
