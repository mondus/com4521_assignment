#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "stb_image_write.h"
#include "nbody.h"
/**
 * Steal data from visualisation init
 */
extern unsigned int __N;  // Num of particles
extern unsigned int __D;  // Dimensions of histogram
extern const float *Densities;
extern const float *PositionsX;
extern const float *PositionsY;
extern const nbody *Bodies;

#define DIMS_X 1024
#define DIMS_Y 1024
struct Pixel
{
    unsigned char r, g, b;
};
typedef struct Pixel Pixel;
Pixel buff[DIMS_X][DIMS_Y];
void clearImage()
{
    // Set image to black
    memset(buff, 0, sizeof(Pixel)*DIMS_X*DIMS_Y);
}
void renderHistogramToImage()
{
    if(Densities) {
        // Debug, overwrite hist
        //for (unsigned int i = 0; i < __D * __D; ++i)
        //{
        //    histogram[i] = i / (float)(__D*__D);
        //}
        float _D_X = DIMS_X / (float)__D;
        float _D_Y = DIMS_Y / (float)__D;
        // Iterate histogram cells
        for (unsigned int i = 0; i < __D; ++i) {
            const unsigned int min_y = (unsigned int)round(i * _D_Y);
            const unsigned int max_y = (unsigned int)round((i + 1) * _D_Y);
            for (unsigned int j = 0; j < __D; ++j) {
                // Calc image x/y area
                const unsigned int min_x = (unsigned int)round(j * _D_X);
                const unsigned int max_x = (unsigned int)round((j + 1) * _D_X);
                for (unsigned int y = min_y; y < max_y; ++y) {
                    //const unsigned int offset_y = y * DIMS_X;
                    for (unsigned int x = min_x; x < max_x; ++x) {
                        //const unsigned int offset = offset_y + x;
                        buff[y][x].r = (unsigned char)(Densities[i * __D + j] * UCHAR_MAX);
                    }
                }
            }
        }
    } else {
         printf("Error: You setHistogramData() or setActivityMapData() must be called before renderHistogramToImage()\n");
    }
}
void renderNBodyToImage()
{
    if (Bodies) {
        for (unsigned int i = 0; i < __N; ++i) {
            const unsigned int x = (unsigned int)(DIMS_X * Bodies[i].x);
            const unsigned int y = (unsigned int)(DIMS_Y * Bodies[i].y);
            if (x < DIMS_X && y < DIMS_Y) {
                buff[y][x].r = UCHAR_MAX;
                buff[y][x].g = UCHAR_MAX;
                buff[y][x].b = UCHAR_MAX;
            }
        }
    } else if(PositionsX && PositionsY) {
        for (unsigned int i = 0; i < __N; ++i) {
            const unsigned int x = (unsigned int)(DIMS_X * Bodies[i].x);
            const unsigned int y = (unsigned int)(DIMS_Y * Bodies[i].y);
            if (x < DIMS_X && y < DIMS_Y) {
                buff[y][x].r = UCHAR_MAX;
                buff[y][x].g = UCHAR_MAX;
                buff[y][x].b = UCHAR_MAX;
            }
        }
    } else {
        printf("Error: You setNBodyPositions2f() or setNBodyPositions() must be called before renderNBodyToImage()\n");
    }
}
extern int mkdir(const char *);
void writeImage(unsigned int id)
{
static int hasRan = 1;
if (hasRan) {
#ifdef _MSC_VER
    mkdir("out");
#else
    mkdir("out", 777);
#endif
    hasRan = 0;
}
    char name_buffer[1024];
    memset(name_buffer, 0, sizeof(char) * 1024);
    sprintf(name_buffer, "out/frame_%u.png", id);

    if (!buff || !stbi_write_png(name_buffer, DIMS_X, DIMS_Y, 3, buff, 0))
        fprintf(stderr, "Failed to write image %u\n", id);
}
