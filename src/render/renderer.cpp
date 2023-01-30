#include "renderer.h"
#include <algorithm>
#include <functional>
#include <glm/gtx/component_wise.hpp>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

namespace render {

// The renderer is passed a pointer to the volume, gradient volume, camera and an initial renderConfig.
// The camera being pointed to may change each frame (when the user interacts). When the renderConfig
// changes the setConfig function is called with the updated render config. This gives the Renderer an
// opportunity to resize the framebuffer.
Renderer::Renderer(
    const volume::Volume* pVolume,
    const volume::GradientVolume* pGradientVolume,
    const render::RayTraceCamera* pCamera,
    const RenderConfig& initialConfig)
    : m_pVolume(pVolume)
    , m_pGradientVolume(pGradientVolume)
    , m_pCamera(pCamera)
    , m_config(initialConfig)
{
    resizeImage(initialConfig.renderResolution);
}

// Set a new render config if the user changed the settings.
void Renderer::setConfig(const RenderConfig& config)
{
    if (config.renderResolution != m_config.renderResolution)
        resizeImage(config.renderResolution);
    m_config = config;
}

// Resize the framebuffer and fill it with black pixels.
void Renderer::resizeImage(const glm::ivec2& resolution)
{
    m_frameBuffer.resize(size_t(resolution.x) * size_t(resolution.y), glm::vec4(0.0f));
}

// Clear the framebuffer by setting all pixels to black.
void Renderer::resetImage()
{
    std::fill(std::begin(m_frameBuffer), std::end(m_frameBuffer), glm::vec4(0.0f));
}

// Return a VIEW into the framebuffer. This view is merely a reference to the m_frameBuffer member variable.
// This does NOT make a copy of the framebuffer.
gsl::span<const glm::vec4> Renderer::frameBuffer() const
{
    return m_frameBuffer;
}

// Main render function. It computes an image according to the current renderMode.
// Multithreading is enabled in Release/RelWithDebInfo modes. In Debug mode multithreading is disabled to make debugging easier.
void Renderer::render()
{
    resetImage();

    static constexpr float sampleStep = 1.0f;
    const glm::vec3 planeNormal = -glm::normalize(m_pCamera->forward());
    const glm::vec3 volumeCenter = glm::vec3(m_pVolume->dims()) / 2.0f;
    const Bounds bounds { glm::vec3(0.0f), glm::vec3(m_pVolume->dims() - glm::ivec3(1)) };

    // 0 = sequential (single-core), 1 = TBB (multi-core)
#ifdef NDEBUG
    // If NOT in debug mode then enable parallelism using the TBB library (Intel Threaded Building Blocks).
#define PARALLELISM 1
#else
    // Disable multi threading in debug mode.
#define PARALLELISM 0
#endif

#if PARALLELISM == 0
    // Regular (single threaded) for loops.
    for (int x = 0; x < m_config.renderResolution.x; x++) {
        for (int y = 0; y < m_config.renderResolution.y; y++) {
#else
    // Parallel for loop (in 2 dimensions) that subdivides the screen into tiles.
    const tbb::blocked_range2d<int> screenRange { 0, m_config.renderResolution.y, 0, m_config.renderResolution.x };
    tbb::parallel_for(screenRange, [&](tbb::blocked_range2d<int> localRange) {
        // Loop over the pixels in a tile. This function is called on multiple threads at the same time.
        for (int y = std::begin(localRange.rows()); y != std::end(localRange.rows()); y++) {
            for (int x = std::begin(localRange.cols()); x != std::end(localRange.cols()); x++) {
#endif
            // Compute a ray for the current pixel.
            const glm::vec2 pixelPos = glm::vec2(x, y) / glm::vec2(m_config.renderResolution);
            Ray ray = m_pCamera->generateRay(pixelPos * 2.0f - 1.0f);

            // Compute where the ray enters and exists the volume.
            // If the ray misses the volume then we continue to the next pixel.
            if (!intersectRayVolumeBounds(ray, bounds))
                continue;

            // Get a color for the current pixel according to the current render mode.
            glm::vec4 color {};
            switch (m_config.renderMode) {
            case RenderMode::RenderSlicer: {
                color = traceRaySlice(ray, volumeCenter, planeNormal);
                break;
            }
            case RenderMode::RenderMIP: {
                m_config.shadingMode = ShadingMode::Phong;
                color = traceRayMIP(ray, sampleStep);
                break;
            }
            case RenderMode::RenderComposite: {
                color = traceRayComposite(ray, sampleStep);
                break;
            }
            case RenderMode::RenderIso: {
                color = traceRayISO(ray, sampleStep);
                break;
            }
            case RenderMode::RenderTF2D: {
                color = traceRayTF2D(ray, sampleStep);
                break;
            }
            }
            // Write the resulting color to the screen.
            fillColor(x, y, color);

#if PARALLELISM == 1
        }
    }
});
#else
            }
        }
#endif
}

// This function returns the color of given pixel based on the current shading mode
glm::vec3 Renderer::computeShading(const glm::vec3& color, const volume::GradientVoxel& gradient, const glm::vec3& L, const glm::vec3& V, const glm::vec3& samplePos) const
{

    switch (m_config.shadingMode) {
    case ShadingMode::Phong: {
        return computePhongShading(color, gradient, L, V);
    }
    case ShadingMode::Technical: {
        return computeTechnicalShading(gradient, L, V);
    }
    case ShadingMode::Normal:
        return computeNormalShading(gradient, L, V, samplePos);
    default: {
        throw std::exception();
    }
    }
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// This function generates a view alongside a plane perpendicular to the camera through the center of the volume
//  using the slicing technique.
glm::vec4 Renderer::traceRaySlice(const Ray& ray, const glm::vec3& volumeCenter, const glm::vec3& planeNormal) const
{
    const float t = glm::dot(volumeCenter - ray.origin, planeNormal) / glm::dot(ray.direction, planeNormal);
    const glm::vec3 samplePos = ray.origin + ray.direction * t;
    const float val = m_pVolume->getSampleInterpolate(samplePos);
    return glm::vec4(glm::vec3(std::max(val / m_pVolume->maximum(), 0.0f)), 1.f);
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// Function that implements maximum-intensity-projection (MIP) raycasting.
// It returns the color assigned to a ray/pixel given its origin, direction and the distances
// at which it enters/exits the volume (ray.tmin & ray.tmax respectively).
// The ray must be sampled with a distance defined by the sampleStep
glm::vec4 Renderer::traceRayMIP(const Ray& ray, float sampleStep) const
{
    float maxVal = 0.0f;

    // Incrementing samplePos directly instead of recomputing it each frame gives a measurable speed-up.
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;
    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        const float val = m_pVolume->getSampleInterpolate(samplePos);
        maxVal = std::max(val, maxVal);
    }

    // Normalize the result to a range of [0 to mpVolume->maximum()].
    return glm::vec4(glm::vec3(maxVal) / m_pVolume->maximum(), 1.0f);
}

// This function should find the position where the ray intersects with the volume's isosurface.
// If volume shading is DISABLED then simply return the isoColor.
// If volume shading is ENABLED then return the phong-shaded color at that location using the local gradient (from m_pGradientVolume).
//   Use the camera position (m_pCamera->position()) as the light position.
// Use the bisectionAccuracy function to get a more precise isosurface location between two steps.
glm::vec4 Renderer::traceRayISO(const Ray& ray, float sampleStep) const
{
    static constexpr glm::vec3 isoColor { 0.8f, 0.8f, 0.2f };
    static constexpr glm::vec3 noIntersectionColor { 0.0f, 0.0f, 0.0f };
    static constexpr glm::vec3 backgroundColor { 1.0f, 1.0f, 1.0f };

    glm::vec3 finalPixelColor = noIntersectionColor;
    if (m_config.shadingMode == ShadingMode::Normal)
        finalPixelColor = backgroundColor;

    float lastIterationT = std::numeric_limits<float>::lowest();
    float isoValue = m_config.isoValue;

    // Incrementing samplePos directly instead of recomputing it each frame gives a measurable speed-up.
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;
    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        const float val = m_pVolume->getSampleInterpolate(samplePos);

        if (val > isoValue) {
            float accurateT = bisectionAccuracy(ray, lastIterationT, t, isoValue);

            if (m_config.volumeShading) {
                finalPixelColor = computeShading(isoColor,
                    m_pGradientVolume->getGradientInterpolate(ray.origin + accurateT * ray.direction),
                    m_pCamera->position(),
                    m_pCamera->position(), samplePos);
            } else {
                finalPixelColor = isoColor;
            }
            break;
        }
        lastIterationT = t;
    }

    return glm::vec4(finalPixelColor, 1.0f);
}

// Given that the iso value lies somewhere between t0 and t1, find a t for which the value
// closely matches the iso value (less than 0.01 difference). Add a limit to the number of
// iterations such that it does not get stuck in degenerate cases.
float Renderer::bisectionAccuracy(const Ray& ray, float t0, float t1, float isoValue) const
{
    static const float difference = 0.01f;
    float t = (t0 + t1) / 2;
    float t_previous;
    int iteration = 0;
    int max_iterations = 100;
    float value;

    while (iteration < max_iterations) {
        glm::vec3 samplePos = ray.origin + t * ray.direction;
        value = m_pVolume->getSampleInterpolate(samplePos);

        if (glm::abs(value - isoValue) < difference) {
            return t;
        } else if (value < isoValue) {
            t0 = t;
        } else {
            t1 = t;
        }

        t_previous = t;
        t = (t0 + t1) / 2;
        if (t == t_previous) {
            return t;
        }

        iteration++;
    }
    return t;
}

// Compute Phong Shading given the voxel color (material color), the gradient, the light vector and view vector.
// You can find out more about the Phong shading model at:
// https://en.wikipedia.org/wiki/Phong_reflection_model
//
// Use the given color for the ambient/specular/diffuse (you are allowed to scale these constants by a scalar value).
// You are free to choose any specular power that you'd like.
glm::vec3 Renderer::computePhongShading(const glm::vec3& color, const volume::GradientVoxel& gradient, const glm::vec3& L, const glm::vec3& V) const
{
    static constexpr float kA = 0.1f;
    static constexpr float kD = 0.7f;
    static constexpr float kS = 0.2f;
    static constexpr int alpha = 2.0;

    static constexpr glm::vec3 IA = glm::vec3(1.0f);
    static constexpr glm::vec3 ID = glm::vec3(1.0f);
    static constexpr glm::vec3 IS = glm::vec3(1.0f);

    glm::vec3 Nn = glm::normalize(gradient.dir);
    glm::vec3 Ln = glm::normalize(L);
    glm::vec3 Vn = glm::normalize(V);

    glm::vec3 Rn = glm::normalize(glm::reflect(-Ln, Nn));

    glm::vec3 ambient = IA * kA * color;
    glm::vec3 diffuse = ID * kD * glm::max(0.0f, glm::dot(Ln, Vn)) * color;
    glm::vec3 specular = IS * kS * static_cast<float>(glm::pow(glm::max(0.0f, glm::dot(Rn, Vn)), alpha)) * color;

    return ambient + diffuse + specular;
}

glm::vec3 Renderer::computeTechnicalShading(const volume::GradientVoxel& gradient, const glm::vec3& L, const glm::vec3& V) const
{
    float beta = m_config.beta;
    float alpha = m_config.alpha;
    float b = m_config.b;
    float y = m_config.y;
    glm::vec3 kBlue = glm::vec3(0, 0, b);
    glm::vec3 kYellow = glm::vec3(y, y, 0);
    float diffuse = glm::dot(glm::normalize(gradient.dir), glm::normalize(L));
    diffuse = (diffuse * 0.5f + 0.5f) * 1.0f;
    float kD = 1.0f;
    glm::vec3 kCool = kBlue + alpha * kD;
    glm::vec3 kWarm = kYellow + beta * kD;
    glm::vec3 diff = glm::vec3(1.0f) * (diffuse * kWarm + (1 - diffuse) * kCool);
    glm::vec3 specular = glm::vec3(1.0f) * glm::pow(glm::max(0.0f, glm::dot(glm::normalize(gradient.dir), glm::normalize(L + V))), alpha);
    glm::vec3 finalColor = diff + specular;
    return finalColor;
}

glm::vec3 Renderer::computeNormalShading(const volume::GradientVoxel& gradient, const glm::vec3& L, const glm::vec3& V, const glm::vec3& samplePos) const
{
    float near = 0.1;
    float far = 100.0;

    float z = samplePos.z * 2.0 - 1.0;
    float linearizedDepth = (2.0 * near * far) / (far + near - z * (far - near));
    glm::vec3 depth = glm::vec3(linearizedDepth / far);
    return depth;
}

// In this function, implement 1D transfer function raycasting.
// Use getTFValue to compute the color for a given volume value according to the 1D transfer function.
glm::vec4 Renderer::traceRayComposite(const Ray& ray, float sampleStep) const
{
    const glm::vec3 increment = sampleStep * ray.direction;
    glm::vec3 accumulatedOpacity(0.0f);
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    float opacity = 0.0f;

    // loops until the ray is at infinite or the opacity has reached its maximum
    for (float t = ray.tmin; t <= ray.tmax && opacity < 1.0f; t += sampleStep, samplePos += increment) {
        float intensity = m_pVolume->getSampleInterpolate(samplePos);
        glm::vec4 sampleOpacity = getTFValue(intensity);

        updateRayOpacity(sampleOpacity, accumulatedOpacity, opacity);
    }

    if (m_config.volumeShading) {
        accumulatedOpacity *= computeShading(
            accumulatedOpacity,
            m_pGradientVolume->getGradientInterpolate(samplePos),
            m_pCamera->position(),
            m_pCamera->position(),
            samplePos);
    }
    return glm::vec4(accumulatedOpacity, opacity);
}

// This function takes the accumulated opacity, the current ray position opacity and updates all the
// color channels and the opacity itself
void Renderer::updateRayOpacity(const glm::vec4& sampleOpacity, glm::vec3& accumulatedOpacity, float& opacity) const
{
    accumulatedOpacity.r = accumulatedOpacity.r + (1.0f - opacity) * sampleOpacity.a * sampleOpacity.r;
    accumulatedOpacity.g = accumulatedOpacity.g + (1.0f - opacity) * sampleOpacity.a * sampleOpacity.g;
    accumulatedOpacity.b = accumulatedOpacity.b + (1.0f - opacity) * sampleOpacity.a * sampleOpacity.b;
    opacity = opacity + (1.0f - opacity) * sampleOpacity.a;
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// Looks up the color+opacity corresponding to the given volume value from the 1D transfer function LUT (m_config.tfColorMap).
// The value will initially range from (m_config.tfColorMapIndexStart) to (m_config.tfColorMapIndexStart + m_config.tfColorMapIndexRange) .
glm::vec4 Renderer::getTFValue(float val) const
{
    // Map value from [m_config.tfColorMapIndexStart, m_config.tfColorMapIndexStart + m_config.tfColorMapIndexRange) to [0, 1) .
    const float range01 = (val - m_config.tfColorMapIndexStart) / m_config.tfColorMapIndexRange;
    const size_t i = std::min(static_cast<size_t>(range01 * static_cast<float>(m_config.tfColorMap.size())), m_config.tfColorMap.size() - 1);
    return m_config.tfColorMap[i];
}

// In this function, implement 2D transfer function raycasting.
// Use the getTF2DOpacity function that you implemented to compute the opacity according to the 2D transfer function.
glm::vec4 Renderer::traceRayTF2D(const Ray& ray, float sampleStep) const
{
    const glm::vec3 increment = sampleStep * ray.direction;
    glm::vec3 accumulatedOpacity(0.0f);
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    float opacity = 0.0f;

    // loops until the ray is at infinite or the opacity has reached its maximum
    // front-to-back compositing as per slide 81, Lecture 06
    for (float t = ray.tmin; t <= ray.tmax && opacity < 1.0f; t += sampleStep, samplePos += increment) {
        float intensity = m_pVolume->getSampleInterpolate(samplePos);
        float sampleOpacity = getTF2DOpacity(intensity, m_pGradientVolume->getGradientInterpolate(samplePos).magnitude);

        updateRay2DOpacity(sampleOpacity, accumulatedOpacity, opacity);
    }

    if (m_config.volumeShading) {
        accumulatedOpacity *= computeShading(
            accumulatedOpacity,
            m_pGradientVolume->getGradientInterpolate(samplePos),
            m_pCamera->position(),
            m_pCamera->position(),
            samplePos);
    }
    return glm::vec4(accumulatedOpacity, opacity);
}

void Renderer::updateRay2DOpacity(const float& sampleOpacity, glm::vec3& accumulatedOpacity, float& opacity) const
{
    // as per slide 82 in Lecture 06
    accumulatedOpacity.r = accumulatedOpacity.r + (1.0f - opacity) * sampleOpacity * m_config.TF2DColor.r;
    accumulatedOpacity.g = accumulatedOpacity.g + (1.0f - opacity) * sampleOpacity * m_config.TF2DColor.g;
    accumulatedOpacity.b = accumulatedOpacity.b + (1.0f - opacity) * sampleOpacity * m_config.TF2DColor.b;
    opacity = opacity + (1.0f - opacity) * sampleOpacity;
}

// This function should return an opacity value for the given intensity and gradient according to the 2D transfer function.
// Calculate whether the values are within the radius/intensity triangle defined in the 2D transfer function widget.
// If so: return a tent weighting as described in the assignment
// Otherwise: return 0.0f
//
// The 2D transfer function settings can be accessed through m_config.TF2DIntensity and m_config.TF2DRadius.
float Renderer::getTF2DOpacity(float intensity, float gradientMagnitude) const
{
    float opacity = 0.0f;
    float gradientUpperBound = m_pGradientVolume->maxMagnitude() / m_config.TF2DRadius * std::abs(intensity - m_config.TF2DIntensity);

    if (gradientMagnitude >= gradientUpperBound) {
        float slope = m_config.TF2DColor.a / (m_config.TF2DRadius * m_pGradientVolume->maxMagnitude());
        opacity = slope * gradientMagnitude + (m_config.TF2DIntensity - slope * m_pGradientVolume->maxMagnitude() / 2);
        opacity = std::min(opacity, m_config.TF2DColor.a);
    }

    return opacity;
}

// This function computes if a ray intersects with the axis-aligned bounding box around the volume.
// If the ray intersects then tmin/tmax are set to the distance at which the ray hits/exists the
// volume and true is returned. If the ray misses the volume the function returns false.
//
// If you are interested you can learn about it at.
// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
bool Renderer::intersectRayVolumeBounds(Ray& ray, const Bounds& bounds) const
{
    const glm::vec3 invDir = 1.0f / ray.direction;
    const glm::bvec3 sign = glm::lessThan(invDir, glm::vec3(0.0f));

    float tmin = (bounds.lowerUpper[sign[0]].x - ray.origin.x) * invDir.x;
    float tmax = (bounds.lowerUpper[!sign[0]].x - ray.origin.x) * invDir.x;
    const float tymin = (bounds.lowerUpper[sign[1]].y - ray.origin.y) * invDir.y;
    const float tymax = (bounds.lowerUpper[!sign[1]].y - ray.origin.y) * invDir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;
    tmin = std::max(tmin, tymin);
    tmax = std::min(tmax, tymax);

    const float tzmin = (bounds.lowerUpper[sign[2]].z - ray.origin.z) * invDir.z;
    const float tzmax = (bounds.lowerUpper[!sign[2]].z - ray.origin.z) * invDir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    ray.tmin = std::max(tmin, tzmin);
    ray.tmax = std::min(tmax, tzmax);
    return true;
}

// This function inserts a color into the framebuffer at position x,y
void Renderer::fillColor(int x, int y, const glm::vec4& color)
{
    const size_t index = static_cast<size_t>(m_config.renderResolution.x * y + x);
    m_frameBuffer[index] = color;
}
}