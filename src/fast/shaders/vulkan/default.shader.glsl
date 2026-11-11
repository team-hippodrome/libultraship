@prism(type='fragment', name='Fast3D Vulkan Shader', version='1.0.0', description='Ported shader to prism', author='Emill & Prism Team')

#version 450

// PORT: Vulkan backend (gfx_vulkan.cpp). The combiner follows default.shader.metal (wrap-around WRAP, noise,
// dual-source output); the GLSL is the OpenGL template's. Vertex attributes and varyings have fixed locations:
// 0 position, 1-3 texture 0 (coordinates, clamp S, clamp T), 4-6 texture 1, 7 fog, 8 grayscale, 9.. inputs.
// The vertex buffer packs them in that order (CreateAndLoadNewShader builds the matching attribute list).
// (loc_tex and loc_clamp hold the texture locations: prism's arithmetic has no operator precedence)

@if(VERTEX_SHADER)
    layout(location = 0) in vec4 aVtxPos;

    @for(i in 0..2)
        @if(o_textures[i])
            layout(location = @{loc_tex[i]}) in vec2 aTexCoord@{i};
            layout(location = @{loc_tex[i]}) out vec2 vTexCoord@{i};
            @for(j in 0..2)
                @if(o_clamp[i][j])
                    @if(j == 0)
                        layout(location = @{loc_clamp[i][0]}) in float aTexClampS@{i};
                        layout(location = @{loc_clamp[i][0]}) out float vTexClampS@{i};
                    @else
                        layout(location = @{loc_clamp[i][1]}) in float aTexClampT@{i};
                        layout(location = @{loc_clamp[i][1]}) out float vTexClampT@{i};
                    @end
                @end
            @end
        @end
    @end

    @if(o_fog)
        layout(location = 7) in vec4 aFog;
        layout(location = 7) out vec4 vFog;
    @end

    @if(o_grayscale)
        layout(location = 8) in vec4 aGrayscaleColor;
        layout(location = 8) out vec4 vGrayscaleColor;
    @end

    @for(i in 0..o_inputs)
        @if(o_alpha)
            layout(location = @{9 + i}) in vec4 aInput@{i + 1};
            layout(location = @{9 + i}) out vec4 vInput@{i + 1};
        @else
            layout(location = @{9 + i}) in vec3 aInput@{i + 1};
            layout(location = @{9 + i}) out vec3 vInput@{i + 1};
        @end
    @end

    void main() {
        @for(i in 0..2)
            @if(o_textures[i])
                vTexCoord@{i} = aTexCoord@{i};
                @for(j in 0..2)
                    @if(o_clamp[i][j])
                        @if(j == 0)
                            vTexClampS@{i} = aTexClampS@{i};
                        @else
                            vTexClampT@{i} = aTexClampT@{i};
                        @end
                    @end
                @end
            @end
        @end
        @if(o_fog)
            vFog = aFog;
        @end
        @if(o_grayscale)
            vGrayscaleColor = aGrayscaleColor;
        @end
        @for(i in 0..o_inputs)
            vInput@{i + 1} = aInput@{i + 1};
        @end
        gl_Position = aVtxPos;
    }
@else
    @if(dual_out)
    // PORT: dual-source blending: the colour blends by vOutColor1.a; the alpha written is vOutColor.a (coverage)
    layout(location = 0, index = 0) out vec4 vOutColor;
    layout(location = 0, index = 1) out vec4 vOutColor1;
    @end
    @if(single_out)
    layout(location = 0) out vec4 vOutColor;
    @end

    @for(i in 0..2)
        @if(o_textures[i])
            layout(location = @{loc_tex[i]}) in vec2 vTexCoord@{i};
            @for(j in 0..2)
                @if(o_clamp[i][j])
                    @if(j == 0)
                        layout(location = @{loc_clamp[i][0]}) in float vTexClampS@{i};
                    @else
                        layout(location = @{loc_clamp[i][1]}) in float vTexClampT@{i};
                    @end
                @end
            @end
        @end
    @end

    @if(o_fog)
    layout(location = 7) in vec4 vFog;
    @end
    @if(o_grayscale)
    layout(location = 8) in vec4 vGrayscaleColor;
    @end

    @for(i in 0..o_inputs)
        @if(o_alpha)
            layout(location = @{9 + i}) in vec4 vInput@{i + 1};
        @else
            layout(location = @{9 + i}) in vec3 vInput@{i + 1};
        @end
    @end

    // Bindings 0-5: texture 0, texture 1, mask 0, mask 1, blend 0, blend 1 (as the other backends' texture units)
    @if(o_textures[0])
    layout(set = 0, binding = 0) uniform sampler2D uTex0;
    @end
    @if(o_textures[1])
    layout(set = 0, binding = 1) uniform sampler2D uTex1;
    @end
    @if(o_masks[0])
    layout(set = 0, binding = 2) uniform sampler2D uTexMask0;
    @end
    @if(o_masks[1])
    layout(set = 0, binding = 3) uniform sampler2D uTexMask1;
    @end
    @if(o_blend[0])
    layout(set = 0, binding = 4) uniform sampler2D uTexBlend0;
    @end
    @if(o_blend[1])
    layout(set = 0, binding = 5) uniform sampler2D uTexBlend1;
    @end

    // Per-draw constants (VulkanPushConstants in gfx_vulkan.cpp)
    layout(push_constant) uniform PushConstants {
        int frame_count;
        float noise_scale;
        float prim_depth;
        int texture_filtering[2];
    } pc;

    // the names the combiner formulas use (append_formula, shared with the OpenGL backend)
    #define frame_count pc.frame_count
    #define noise_scale pc.noise_scale

    #define TEX_OFFSET(off) texture(tex, texCoord - off / texSize)
    #define WRAP(x, low, high) (mod((x) - (low), (high) - (low)) + (low))

    float random(in vec3 value) {
        float random = dot(sin(value), vec3(12.9898, 78.233, 37.719));
        return fract(sin(random) * 143758.5453);
    }

    vec4 fromLinear(vec4 linearRGB) {
        bvec3 cutoff = lessThan(linearRGB.rgb, vec3(0.0031308));
        vec3 higher = vec3(1.055) * pow(linearRGB.rgb, vec3(1.0 / 2.4)) - vec3(0.055);
        vec3 lower = linearRGB.rgb * vec3(12.92);
        return vec4(mix(higher, lower, cutoff), linearRGB.a);
    }

    vec4 filter3point(in sampler2D tex, in vec2 texCoord, in vec2 texSize) {
        vec2 offset = fract(texCoord * texSize - vec2(0.5));
        offset -= step(1.0, offset.x + offset.y);
        vec4 c0 = TEX_OFFSET(offset);
        vec4 c1 = TEX_OFFSET(vec2(offset.x - sign(offset.x), offset.y));
        vec4 c2 = TEX_OFFSET(vec2(offset.x, offset.y - sign(offset.y)));
        return c0 + abs(offset.x) * (c1 - c0) + abs(offset.y) * (c2 - c0);
    }

    vec4 hookTexture2D(in int filtering, in sampler2D tex, in vec2 uv, in vec2 texSize) {
    @if(o_three_point_filtering)
        if (filtering == @{FILTER_THREE_POINT}) {
            return filter3point(tex, uv, texSize);
        }
    @end
        return texture(tex, uv);
    }

    void main() {
        @for(i in 0..2)
            @if(o_textures[i])
                @{s = o_clamp[i][0]}
                @{t = o_clamp[i][1]}

                vec2 texSize@{i} = vec2(textureSize(uTex@{i}, 0));

                @if(!s && !t)
                    vec2 vTexCoordAdj@{i} = vTexCoord@{i};
                @else
                    @if(s && t)
                        vec2 vTexCoordAdj@{i} = clamp(vTexCoord@{i}, 0.5 / texSize@{i}, vec2(vTexClampS@{i}, vTexClampT@{i}));
                    @elseif(s)
                        vec2 vTexCoordAdj@{i} = vec2(clamp(vTexCoord@{i}.s, 0.5 / texSize@{i}.s, vTexClampS@{i}), vTexCoord@{i}.t);
                    @else
                        vec2 vTexCoordAdj@{i} = vec2(vTexCoord@{i}.s, clamp(vTexCoord@{i}.t, 0.5 / texSize@{i}.t, vTexClampT@{i}));
                    @end
                @end

                vec4 texVal@{i} = hookTexture2D(pc.texture_filtering[@{i}], uTex@{i}, vTexCoordAdj@{i}, texSize@{i});

                @if(o_masks[i])
                    vec2 maskSize@{i} = vec2(textureSize(uTexMask@{i}, 0));
                    vec4 maskVal@{i} = hookTexture2D(pc.texture_filtering[@{i}], uTexMask@{i}, vTexCoordAdj@{i}, maskSize@{i});

                    @if(o_blend[i])
                        vec4 blendVal@{i} = hookTexture2D(pc.texture_filtering[@{i}], uTexBlend@{i}, vTexCoordAdj@{i}, texSize@{i});
                    @else
                        vec4 blendVal@{i} = vec4(0, 0, 0, 0);
                    @end

                    texVal@{i} = mix(texVal@{i}, blendVal@{i}, maskVal@{i}.a);
                @end
            @end
        @end

        @if(o_alpha)
            vec4 texel;
        @else
            vec3 texel;
        @end

        @if(o_2cyc)
            @{f_range = 2}
        @else
            @{f_range = 1}
        @end

        @for(c in 0..f_range)
            @if(c == 1)
                @if(o_alpha)
                    @if(o_c[c][1][2] == SHADER_COMBINED)
                        texel.a = WRAP(texel.a, -1.01, 1.01);
                    @else
                        texel.a = WRAP(texel.a, -0.51, 1.51);
                    @end
                @end

                @if(o_c[c][0][2] == SHADER_COMBINED)
                    texel.rgb = WRAP(texel.rgb, -1.01, 1.01);
                @else
                    texel.rgb = WRAP(texel.rgb, -0.51, 1.51);
                @end
            @end

            @if(!o_color_alpha_same[c] && o_alpha)
                texel = vec4(@{
                append_formula(o_c[c], o_do_single[c][0],
                            o_do_multiply[c][0], o_do_mix[c][0], false, false, true, c == 0)
                }, @{append_formula(o_c[c], o_do_single[c][1],
                            o_do_multiply[c][1], o_do_mix[c][1], true, true, true, c == 0)
                });
            @else
                texel = @{append_formula(o_c[c], o_do_single[c][0],
                            o_do_multiply[c][0], o_do_mix[c][0], o_alpha, false,
                            o_alpha, c == 0)};
            @end
        @end

        @if(o_texture_edge && o_alpha)
            if (texel.a > 0.19) texel.a = 1.0; else discard;
        @end

        texel = WRAP(texel, -0.51, 1.51);
        texel = clamp(texel, 0.0, 1.0);

        @if(o_fog)
            @if(o_alpha)
                texel = vec4(mix(texel.rgb, vFog.rgb, vFog.a), texel.a);
            @else
                texel = mix(texel, vFog.rgb, vFog.a);
            @end
        @end

        @if(o_grayscale)
            float intensity = (texel.r + texel.g + texel.b) / 3.0;
            vec3 new_texel = vGrayscaleColor.rgb * intensity;
            texel.rgb = mix(texel.rgb, new_texel, vGrayscaleColor.a);
        @end

        @if(o_alpha && o_noise)
            vec2 coords = gl_FragCoord.xy * noise_scale;
            texel.a *= round(clamp(random(vec3(floor(coords), float(frame_count))) + texel.a - 0.5, 0.0, 1.0));
        @end

        @if(o_alpha)
            @if(o_alpha_threshold)
                if (texel.a < 8.0 / 256.0) discard;
            @end
            // Depth-only masks preserve colour and coverage. The second output's alpha drives RGB blending.
            @if(o_invisible)
                texel.a = 0.0;
            @end
            @if(dual_out)
                // PORT: the alpha written is the N64 coverage bit: 1 for every drawn pixel, kept (0) for o_invisible
                // draws; the blend alpha goes to the second output (as default.shader.metal)
                @if(o_invisible)
                    vOutColor = vec4(texel.rgb, 0.0);
                @else
                    vOutColor = vec4(texel.rgb, 1.0);
                @end
                vOutColor1 = vec4(0.0, 0.0, 0.0, texel.a);
            @else
                vOutColor = texel;
            @end
        @else
            vOutColor = vec4(texel, 1.0);
        @end

        @if(srgb_mode)
            vOutColor = fromLinear(vOutColor);
        @end

        @if(o_prim_depth)
            gl_FragDepth = pc.prim_depth;
        @end
    }
@end
