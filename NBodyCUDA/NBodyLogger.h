#ifndef __NBODYLOGGER_H__
#define __NBODYLOGGER_H__

#include "NBody.h"

/**
 * Resets the cached image to all black
 */
void clearImage();
/**
 * Renders histogram values to cached image
 * @note requires setActivityMapData(const float *) or setHistogramData(const float *) to be called first
 */
void renderHistogramToImage();
/**
 * n body points to cached image
 * @note requires setNBodyPositions2f(const float *, const float *) or setNBodyPositions(const nbody *) to be called first
 */
void renderNBodyToImage();
/**
 * Writes the cached image as png to disk
 * Path: out/frame_<id>.png
 */
void writeImage(unsigned int id);

#endif  // __NBODYLOGGER_H__