#pragma once

// PORT: a game framebuffer's picture survives a size change (window resize). The N64 frame buffers keep their picture
// and menus draw their static background only in a screen's first frames (Free Battle's rule screen), so a new, cleared
// colour texture left the screen black until the next screen. Each backend draws the old picture into the new texture,
// scaled uniformly so that the 4:3 picture (full height, or full width in a window narrower than 4:3) keeps its place;
// what the old texture did not cover stays black. PORT_NO_RESIZE_KEEP=1: off (debug).

#include <algorithm>
#include <cstdlib>

namespace Fast {

inline bool PortResizeKeepEnabled() {
    static const bool sOff = getenv("PORT_NO_RESIZE_KEEP") != nullptr;
    return !sOff;
}

// Where the whole old picture (ow x oh) lands in the new texture (nw x nh), in pixels from the top-left corner. The
// rectangle can reach past the texture (a narrower window): callers clip it or let the draw clip it.
struct PortResizeKeepRect {
    float x0, y0, x1, y1;
};

inline PortResizeKeepRect PortResizeKeepDest(float ow, float oh, float nw, float nh) {
    const float oldPicH = std::min(oh, ow * 3.0f / 4.0f);
    const float newPicH = std::min(nh, nw * 3.0f / 4.0f);
    const float s = newPicH / oldPicH;
    const float x0 = (nw - ow * s) / 2.0f, y0 = (nh - oh * s) / 2.0f;
    return { x0, y0, x0 + ow * s, y0 + oh * s };
}

// The part of the source (ow x oh) and destination rectangles that falls inside the new texture, for copies that do
// not clip (blits): integer pixel rectangles, false when nothing remains.
inline bool PortResizeKeepClipped(int ow, int oh, int nw, int nh, int src[4], int dst[4]) {
    PortResizeKeepRect r = PortResizeKeepDest((float)ow, (float)oh, (float)nw, (float)nh);
    const float sx = (r.x1 - r.x0) / ow, sy = (r.y1 - r.y0) / oh;
    const float cx0 = std::max(r.x0, 0.0f), cy0 = std::max(r.y0, 0.0f);
    const float cx1 = std::min(r.x1, (float)nw), cy1 = std::min(r.y1, (float)nh);
    if (cx1 - cx0 < 1.0f || cy1 - cy0 < 1.0f) {
        return false;
    }
    dst[0] = (int)(cx0 + 0.5f), dst[1] = (int)(cy0 + 0.5f), dst[2] = (int)(cx1 + 0.5f), dst[3] = (int)(cy1 + 0.5f);
    src[0] = (int)((cx0 - r.x0) / sx + 0.5f), src[1] = (int)((cy0 - r.y0) / sy + 0.5f);
    src[2] = (int)((cx1 - r.x0) / sx + 0.5f), src[3] = (int)((cy1 - r.y0) / sy + 0.5f);
    src[2] = std::min(src[2], ow), src[3] = std::min(src[3], oh);
    return true;
}

} // namespace Fast
