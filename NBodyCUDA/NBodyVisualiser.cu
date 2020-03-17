#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

//include the header file 
//Uncomment below if including into a *.c file rather than *.cu
//extern "C" {
	#include "NBodyVisualiser.h"
//}

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define TIMING_FRAME_COUNT 20

//User supplied globals
unsigned int __N;
unsigned int __D;
MODE __M;
const float *PositionsX = nullptr;
const float *PositionsY = nullptr;
const nbody *Bodies = nullptr;
const float *Densities = nullptr;
const float *dPositionsX = nullptr;
const float *dPositionsY = nullptr;
const nbody *dBodies = nullptr;
const float *dDensities = nullptr;
void(*simulate_function)(void) = nullptr;

#ifndef NO_OPENGL

// instancing variables for histogram
GLuint vao_hist = 0;
GLuint vao_hist_vertices = 0;
GLuint tbo_hist = 0;
GLuint tex_hist = 0;
GLuint vao_hist_instance_ids = 0;

// instancing variables for nbody
GLuint vao_nbody = 0;
GLuint vao_nbody_vertices = 0;
GLuint tbo_nbody = 0;
GLuint tex_nbody = 0;
GLuint vao_nbody_instance_ids = 0;


// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_z = 0.0;
float translate_z = -1.0;

// vertex shader handles
GLuint vs_hist_shader = 0;
GLuint vs_nbody_shader = 0;
GLuint vs_hist_program = 0;
GLuint vs_nbody_program = 0;
GLuint vs_hist_instance_index = 0;
GLuint vs_nbody_instance_index = 0;

//render options
bool display_bodies = true;
bool display_denisty = false;

//cuda graphics resources
struct cudaGraphicsResource *cuda_nbody_vbo_resource;
struct cudaGraphicsResource *cuda_hist_vbo_resource;


//timing
float elapsed = 0;
float prev_time = 0;
unsigned int frames;
char title[128];


// function prototypes
void displayLoop(void);
void initHistShader();
void initNBodyShader();
void initHistVertexData();
void initNBodyVertexData();
void initGL();
void destroyViewer();
void render(void);
void checkGLError();
void handleKeyboardDefault(unsigned char key, int x, int y);
void handleMouseDefault(int button, int state, int x, int y);
void handleMouseMotionDefault(int x, int y);
void checkCUDAError(const char *msg);

// Vertex shader source code
const char* hist_vertexShaderSource =
{

    "#version 130				                            							\n"
    "#extension GL_EXT_gpu_shader4 : enable												\n"
	"uniform samplerBuffer instance_tex;												\n"
	"in uint instance_index;						        							\n"
	"void main()																		\n"
	"{																					\n"
    "	float instance_data = texelFetchBuffer(instance_tex, int(instance_index)).x;	\n"
	"	vec4 position = vec4(gl_Vertex.x, gl_Vertex.y, 0.0f, 1.0f);						\n"
	"	gl_FrontColor = vec4(instance_data, 0.0f, 0.0f, 0.0f);							\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * position;		    				\n"
	"}																					\n"
};
const char* nbody_vertexShaderSource =
{
    "#version 130										                                \n"
    "#extension GL_EXT_gpu_shader4 : enable												\n"
	"uniform samplerBuffer instance_tex;												\n"
	"in uint instance_index;									        				\n"
	"void main()																		\n"
	"{																					\n"
    "	vec2 instance_data = texelFetchBuffer(instance_tex, int(instance_index)).xy;	\n"
	"	vec4 position = vec4(gl_Vertex.x+instance_data.x,								\n"
	"					     gl_Vertex.y+instance_data.y,								\n"
	"						 gl_Vertex.z, 1.0f);										\n"
	"	gl_FrontColor = vec4(1.0f, 1.0f, 1.0f, 0.0f);									\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * position;		    				\n"
	"}																					\n"
};

//////////////////////////////// CUDA Kernels              ////////////////////////////////

__global__ void copyNBodyData2f(float* buffer, const float *x, const float *y, unsigned int N)
{
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;

	if (i < N){
		//copy data to mapped buffer
		float* ptr = &buffer[i * 2];
		ptr[0] = x[i];
		ptr[1] = y[i];
	}
}


__global__ void copyNBodyData(float* buffer, const nbody* bodies, unsigned int N)
{
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;

	if (i < N){
		//copy data to mapped buffer
		float* ptr = &buffer[i * 2];
		ptr[0] = bodies[i].x;
		ptr[1] = bodies[i].y;
	}
}

__global__ void copyHistData(float* buffer, const float* densities, unsigned int D)
{
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;

	if (i < D*D){
		//copy data to mapped buffer
		buffer[i] = densities[i];
	}
}
#else
#include "NBodyLogger.h"

void destroyViewer()
{
    freeVisualisationMemory();
}
#endif

//////////////////////////////// Header declared functions ////////////////////////////////
void initViewer(unsigned int n, unsigned int d, MODE m, void(*simulate)(void))
{
	__N = n;
	__D = d;
	__M = m;
	simulate_function = simulate;
#ifndef NO_OPENGL
	//check for UVA (not available in 32 bit host mode)
	if (__M == CUDA){
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		if (prop.unifiedAddressing != 1){
			printf("Error: No UVA found. Are you trying to build you CUDA code in 32bit mode?\n");
		}
	}

	//initialiser the open gl viewer and context
	initGL();

	//init out instance rendering and the data
	initHistShader();
	initNBodyShader();
	initHistVertexData();
	initNBodyVertexData();
#endif

}

void setNBodyPositions2f(const float *positions_x, const float *positions_y)
{
	//check that the supplied pointers are device pointers
	if (__M == CUDA){
		cudaPointerAttributes attributes;

		//host allocated memory will cause an error
		if (cudaPointerGetAttributes(&attributes, positions_x) == cudaErrorInvalidValue){
			cudaGetLastError(); // clear out the previous API error
			printf("Error: Pointer (positions_x) passed to setNBodyPositions2f must be a device pointer in CUDA mode!\n");
			return;
		}

		//memory allocated by the device will not be the result may be cudaMemoryTypeHost if UVA was used.
		if (!attributes.devicePointer){
			printf("Error: Pointer (positions_x) passed to setNBodyPositions2f must be a device pointer in CUDA mode!\n");
			return;
		}

		//host allocated memory will cause an error
		if (cudaPointerGetAttributes(&attributes, positions_y) == cudaErrorInvalidValue){
			cudaGetLastError(); // clear out the previous API error
			printf("Error: Pointer (positions_y) passed to setNBodyPositions2f must be a device pointer in CUDA mode!\n");
			return;
		}

		//memory allocated by the device will not be the result may be cudaMemoryTypeHost if UVA was used.
		if (!attributes.devicePointer){
			printf("Error: Pointer (positions_y) passed to setNBodyPositions2f must be a device pointer in CUDA mode!\n");
			return;
		}
	}
#ifndef NO_OPENGL
	PositionsX = positions_x;
	PositionsY = positions_y;
	if (Bodies != 0){
		printf("Warning: You should use either setNBodyPositions2f or setNBodyPositions\n");
	}
#else
    if (__M == CUDA) {
        dPositionsX = positions_x;
        dPositionsY = positions_y;
        if (dBodies != 0) {
            printf("Warning: You should use either setNBodyPositions2f or setNBodyPositions\n");
        }
}
    else {
        PositionsX = positions_x;
        PositionsY = positions_y;
        if (Bodies != 0) {
            printf("Warning: You should use either setNBodyPositions2f or setNBodyPositions\n");
        }
    }
#endif
}

void setNBodyPositions(const nbody *bodies)
{
	//check that the supplied pointer is a device pointer
	if (__M == CUDA){
		cudaPointerAttributes attributes;

		//host allocated memory will cause an error
		if (cudaPointerGetAttributes(&attributes, bodies) == cudaErrorInvalidValue){
			cudaGetLastError(); // clear out the previous API error
			printf("Error: Pointer (bodies) passed to setNBodyPositions must be a device pointer in CUDA mode!\n");
			return;
		}

		//memory allocated by the device will not be the result may be cudaMemoryTypeHost if UVA was used.
		if (!attributes.devicePointer){
			printf("Error: Pointer (bodies) passed to setNBodyPositions must be a device pointer in CUDA mode!\n");
			return;
		}
	}

#ifndef NO_OPENGL
	Bodies = bodies;
	if ((PositionsX != 0) || (PositionsY != 0)){
		printf("Warning: You should use either setNBodyPositions2f or setNBodyPositions\n");
	}
#else
    if (__M == CUDA) {
        dBodies = bodies;
        if ((dPositionsX != 0) || (dPositionsY != 0)) {
            printf("Warning: You should use either setNBodyPositions2f or setNBodyPositions\n");
        }
    } else {
        Bodies = bodies;
        if ((PositionsX != 0) || (PositionsY != 0)) {
            printf("Warning: You should use either setNBodyPositions2f or setNBodyPositions\n");
        }
    }
#endif
}

void setHistogramData(const float *densities)
{
	setActivityMapData(densities);
}

void setActivityMapData(const float *activity)
{
	//if CUDA check that the supplied pointer is a device pointer
	if (__M == CUDA){
		cudaPointerAttributes attributes;

		//host allocated memory will cause an error
		if (cudaPointerGetAttributes(&attributes, activity) == cudaErrorInvalidValue){
			cudaGetLastError(); // clear out the previous API error
			printf("Error: Pointer passed to setActivityMap (or setHistogramData) must be a device pointer in CUDA mode!\n");
			return;
		}

		//memory allocated by the device will not be the result may be cudaMemoryTypeHost if UVA was used.
		if (!attributes.devicePointer){
			printf("Error: Pointer passed to setActivityMap (or setHistogramData) must be a device pointer in CUDA mode!\n");
			return;
		}
	}

	Densities = activity;
#ifndef NO_OPENGL
#else
    //if CUDA check that the supplied pointer is a device pointer
    if (__M == CUDA) {
        cudaPointerAttributes attributes;

        //host allocated memory will cause an error
        if (cudaPointerGetAttributes(&attributes, activity) == cudaErrorInvalidValue) {
            cudaGetLastError(); // clear out the previous API error
            printf("Error: Pointer passed to setActivityMap (or setHistogramData) must be a device pointer in CUDA mode!\n");
            return;
        }

        //memory allocated by the device will not be the result may be cudaMemoryTypeHost if UVA was used.
        if (!attributes.devicePointer) {
            printf("Error: Pointer passed to setActivityMap (or setHistogramData) must be a device pointer in CUDA mode!\n");
            return;
        }
        dDensities = activity;
    }
    else {
        Densities = activity;
    }
#endif
}

void startVisualisationLoop()
{
#ifndef NO_OPENGL
	glutMainLoop();
#else
    unsigned int i = 0;
    while (1)
    {
        //call the simulation function
        simulate_function();
        if (i % 50 == 0) {
            // Log the image
            clearImage();
            renderHistogramToImage();
            renderNBodyToImage();
            writeImage(i);
        }
        i++;
        // Best not to let it run forever writing to disk
        if (i > 100000)
            break;
    }
#endif
}

//////////////////////////////// Source module functions ////////////////////////////////

#ifndef NO_OPENGL
void displayLoop(void)
{
	unsigned int i;
	float *dptr;
	size_t num_bytes;
	unsigned int blocks;
	float t;

	if (simulate_function == 0){
		printf("Error: Simulate function has not been defined by calling initViewer(...)\n");
		return;
	}

	//timing
	if (__M == CUDA)
		cudaDeviceSynchronize();
	t = (float)clock();
	if (prev_time)
		elapsed += t - prev_time;
	prev_time = t;
	frames++;
	if (frames == TIMING_FRAME_COUNT){
		frames = 0;
		elapsed /= TIMING_FRAME_COUNT;
		sprintf(title, "Com4521 Assignment - NBody Visualiser (%f FPS)", 1000.0f /elapsed);
		glutSetWindowTitle(title);
		elapsed = 0;
 	}

	//call the simulation function
	simulate_function();

	//Map data from user supplied pointers into TBO using CUDA
	if (__M == CUDA){
		//NBODY: map buffer to device pointer so Kernel can populate it
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, tbo_nbody);
		num_bytes = __N * 3 * sizeof(float);
		cudaGraphicsMapResources(1, &cuda_nbody_vbo_resource, 0);
		cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_nbody_vbo_resource);
		//kernel to map data into buffer
		blocks = __N / 256;
		if (__N % 256 != 0)
			blocks++;
		//two possible formats for users to supplier body data
		if (Bodies != 0){
			copyNBodyData << <blocks, 256 >> >(dptr, Bodies, __N);
		}
		else if ((PositionsX != 0) && (PositionsY != 0)){
			copyNBodyData2f << <blocks, 256 >> >(dptr, PositionsX, PositionsY, __N);
		}
		cudaGraphicsUnmapResources(1, &cuda_nbody_vbo_resource, 0);
		checkCUDAError("Error copying NBody data from supplier device pointer\n");
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);

		//HIST: map buffer to device pointer so Kernel can populate it
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, tbo_nbody);
		num_bytes = __D*__D * sizeof(float);
		cudaGraphicsMapResources(1, &cuda_hist_vbo_resource, 0);
		cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_hist_vbo_resource);
		//kernel to map data into buffer
		blocks = __D*__D / 256;
		if ((__D*__D) % 256 != 0)
			blocks++;
		copyHistData << <blocks, 256 >> >(dptr, Densities, __D);
		cudaGraphicsUnmapResources(1, &cuda_hist_vbo_resource, 0);
		checkCUDAError("Error copying Activity Map data from supplier device pointer\n");
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);
	}
	//Map data from user supplied pointers into TBO using CPU
	else{
		//map buffer to positions TBO and copy data to it from user supplied pointer
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, tbo_nbody);
		dptr = (float*)glMapBuffer(GL_TEXTURE_BUFFER_EXT, GL_WRITE_ONLY);	//tbo_nbody buffer
		if (dptr == 0){
			printf("Error: Unable to map nBody Texture Buffer Object\n");
			return;
		}
		if (Bodies != 0){
			for (i = 0; i < __N; i++){
				unsigned int index = i * 2;
				dptr[index] = Bodies[i].x;
				dptr[index + 1] = Bodies[i].y;
			}
		}
		else if ((PositionsX != 0) && (PositionsY != 0)){
			for (i = 0; i < __N; i++){
				unsigned int index = i * 2;
				dptr[index] = PositionsX[i];
				dptr[index + 1] = PositionsY[i];
			}
		}
		glUnmapBuffer(GL_TEXTURE_BUFFER_EXT);
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);

		//map hist buffer to positions TBO and copy data to it from user supplied pointer
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, tbo_hist);
		dptr = (float*)glMapBuffer(GL_TEXTURE_BUFFER_EXT, GL_WRITE_ONLY);	//tbo_nbody buffer
		if (dptr == 0){
			printf("Error: Unable to map Histogram Texture Buffer Object\n");
			return;
		}
		if (Densities != 0){
			for (i = 0; i < __D*__D; i++){
				dptr[i] = Densities[i];
			}
		}
		glUnmapBuffer(GL_TEXTURE_BUFFER_EXT);
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);
	}

	//render
	render();
	checkGLError();
}

void initHistShader()
{
	//hist vertex shader
	vs_hist_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs_hist_shader, 1, &hist_vertexShaderSource, 0);
	glCompileShader(vs_hist_shader);


	// check for errors
	GLint status;
	glGetShaderiv(vs_hist_shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Histogram Shader Compilation Error\n");
		char data[1024];
		int len;
		glGetShaderInfoLog(vs_hist_shader, 1024, &len, data);
		printf("%s", data);
	}


	//program
	vs_hist_program = glCreateProgram();
	glAttachShader(vs_hist_program, vs_hist_shader);
	glLinkProgram(vs_hist_program);
	glGetProgramiv(vs_hist_program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Histogram Shader Program Link Error\n");
	}

	glUseProgram(vs_hist_program);

	// get shader variables
	vs_hist_instance_index = glGetAttribLocation(vs_hist_program, "instance_index");
	if (vs_hist_instance_index == (GLuint)-1){
		printf("Warning: Histogram Shader program missing 'attribute in uint instance_index'\n");
	}

	glUseProgram(0);
	//check for any errors
	checkGLError();
}

void initNBodyShader()
{
	//nbody vertex shader
	vs_nbody_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs_nbody_shader, 1,&nbody_vertexShaderSource, 0);
	glCompileShader(vs_nbody_shader);


	// check for errors
	GLint status;
	glGetShaderiv(vs_nbody_shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: nbody Program Shader Compilation Error\n");
		char data[1024];
		int len;
		glGetShaderInfoLog(vs_nbody_shader, 1024, &len, data);
		printf("%s", data);
	}


	//program
	vs_nbody_program = glCreateProgram();
	glAttachShader(vs_nbody_program, vs_nbody_shader);
	glLinkProgram(vs_nbody_program);
	glGetProgramiv(vs_nbody_program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: NBody Shader Program Link Error\n");
	}

	glUseProgram(vs_nbody_program);

	// get shader variables
	vs_nbody_instance_index = glGetAttribLocation(vs_nbody_program, "instance_index");
	if (vs_nbody_instance_index == (GLuint)-1){
		printf("Warning: nbody Program Shader program missing 'attribute in uint instance_index'\n");
	}

	glUseProgram(0);
	//check for any errors
	checkGLError();
}

void initHistVertexData()
{
	/* vertex array object */
	glGenVertexArrays(1, &vao_hist); // Create our Vertex Array Object  
	glBindVertexArray(vao_hist); // Bind our Vertex Array Object so we can use it  

	/* create a vertex buffer */

	// create buffer object (all vertex positions normalised between -0.5 and +0.5)
	glGenBuffers(1, &vao_hist_vertices);
	glBindBuffer(GL_ARRAY_BUFFER, vao_hist_vertices);
	glBufferData(GL_ARRAY_BUFFER, __D*__D * 4 * 3 * sizeof(float), 0, GL_STATIC_DRAW);
	float* verts = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float quad_size = 1.0f / (float)(__D);
	for (unsigned int x = 0; x < __D; x++) {
		for (unsigned int y = 0; y < __D; y++) {
			int offset = (x + (y * (__D))) * 3 * 4;

			float x_min = (float)x / (float)(__D);
			float y_min = (float)y / (float)(__D);

			//first vertex
			verts[offset + 0] = x_min - 0.5f;
			verts[offset + 1] = y_min - 0.5f;
			verts[offset + 2] = 0.0f;

			//second vertex
			verts[offset + 3] = x_min - 0.5f;
			verts[offset + 4] = y_min + quad_size - 0.5f;
			verts[offset + 5] = 0.0f;

			//third vertex
			verts[offset + 6] = x_min + quad_size - 0.5f;
			verts[offset + 7] = y_min + quad_size - 0.5f;
			verts[offset + 8] = 0.0f;

			//fourth vertex
			verts[offset + 9] = x_min + quad_size - 0.5f;
			verts[offset + 10] = y_min - 0.5f;
			verts[offset + 11] = 0.0f;
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glVertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, 0); // Set up our vertex attributes pointer 
	glEnableVertexAttribArray(0);
	checkGLError();

	// instance index buffer
	glGenBuffers(1, &vao_hist_instance_ids);
	glBindBuffer(GL_ARRAY_BUFFER, vao_hist_instance_ids);
	glBufferData(GL_ARRAY_BUFFER, __D*__D * 4 * sizeof(unsigned int), 0, GL_STATIC_DRAW);
	unsigned int* ids = (unsigned int*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	for (unsigned int x = 0; x < __D; x++) {
		for (unsigned int y = 0; y < __D; y++) {
			int index = (x + (y * (__D)));
			int offset = index * 4;

			//four vertices (a quad) have the same instance index
			ids[offset + 0] = index;
			ids[offset + 1] = index;
			ids[offset + 2] = index;
			ids[offset + 3] = index;
		}
	}

	//map instance 
	glVertexAttribIPointer((GLuint)vs_hist_instance_index, 1, GL_UNSIGNED_INT, 0, 0); // Set up instance id attributes pointer in shader
	glEnableVertexAttribArray(vs_hist_instance_index);
	glUnmapBuffer(GL_ARRAY_BUFFER);

	//check for errors
	checkGLError();

	/* texture buffer object */

	glGenBuffers(1, &tbo_hist);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo_hist);
    glBufferData(GL_TEXTURE_BUFFER, __D*__D * 1 * sizeof(float), 0, GL_DYNAMIC_DRAW);		// 1 float elements in a texture buffer object for histogram density

	/* generate texture */
	glGenTextures(1, &tex_hist);
    glBindTexture(GL_TEXTURE_BUFFER, tex_hist);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, tbo_hist);

	//create cuda gl resource to write cuda data to TBO
	if (__M == CUDA){
		cudaGraphicsGLRegisterBuffer(&cuda_hist_vbo_resource, tbo_hist, cudaGraphicsMapFlagsWriteDiscard);
	}

	//unbind buffers
    glBindBuffer(GL_TEXTURE_BUFFER, 0);

	//unbind vao
	glBindVertexArray(0); // Unbind our Vertex Array Object 

	checkGLError();
}

void initNBodyVertexData()
{
	/* vertex array object */
	glGenVertexArrays(1, &vao_nbody);	// Create our Vertex Array Object  
	glBindVertexArray(vao_nbody);		// Bind our Vertex Array Object so we can use it  

	/* create a vertex buffer */

	// create buffer object (all vertex positions normalised between -0.5 and +0.5)
	glGenBuffers(1, &vao_nbody_vertices);
	glBindBuffer(GL_ARRAY_BUFFER, vao_nbody_vertices);
	glBufferData(GL_ARRAY_BUFFER, __N * 3 * sizeof(float), 0, GL_STATIC_DRAW);
	float* verts = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	for (unsigned int i = 0; i < __N; i++) {
			int offset = i*3;

			//vertex point
			verts[offset + 0] = -0.5f;
			verts[offset + 1] = -0.5f;
			verts[offset + 2] = 0.0f;
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glVertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, 0); // Set up our vertex attributes pointer 
	glEnableVertexAttribArray(0);
	checkGLError();

	// instance index buffer
	glGenBuffers(1, &vao_nbody_instance_ids);
	glBindBuffer(GL_ARRAY_BUFFER, vao_nbody_instance_ids);
	glBufferData(GL_ARRAY_BUFFER, __N * 1 * sizeof(unsigned int), 0, GL_STATIC_DRAW);
	unsigned int* ids = (unsigned int*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	for (unsigned int i = 0; i < __N; i++) {
			//single vertex as it is a point
			ids[i] = i;
	}

	//map instance 
	glVertexAttribIPointer((GLuint)vs_nbody_instance_index, 1, GL_UNSIGNED_INT, 0, 0); // Set up instance id attributes pointer in shader
	glEnableVertexAttribArray(vs_nbody_instance_index);
	glUnmapBuffer(GL_ARRAY_BUFFER);

	//check for errors
	checkGLError();

	/* texture buffer object */

	glGenBuffers(1, &tbo_nbody);
	glBindBuffer(GL_TEXTURE_BUFFER, tbo_nbody);
    glBufferData(GL_TEXTURE_BUFFER, __N * 2 * sizeof(float), 0, GL_DYNAMIC_DRAW);		// 2 float elements in a texture buffer object for x and y position

	/* generate texture */
	glGenTextures(1, &tex_nbody);
    glBindTexture(GL_TEXTURE_BUFFER, tex_nbody);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32F, tbo_nbody);

	//create cuda gl resource to write cuda data to TBO
	if (__M == CUDA){
		cudaGraphicsGLRegisterBuffer(&cuda_nbody_vbo_resource, tbo_nbody, cudaGraphicsMapFlagsWriteDiscard);
	}

	//unbind buffers
    glBindBuffer(GL_TEXTURE_BUFFER, 0);

	//unbind vao
	glBindVertexArray(0); // Unbind our Vertex Array Object 

	checkGLError();
}

void destroyViewer()
{
	checkGLError();

	//cleanup hist vao
	glBindVertexArray(vao_hist);
	glDeleteBuffers(1, &vao_hist_vertices);
	vao_hist_vertices = 0;
	glDeleteBuffers(1, &vao_hist_instance_ids);
	vao_hist_instance_ids = 0;
	glDeleteBuffers(1, &tbo_hist);
	tbo_hist = 0;
	glDeleteTextures(1, &tex_hist);
	tex_hist = 0;
	if (__M == CUDA){
		cudaGraphicsUnregisterResource(cuda_hist_vbo_resource);
	}
	glDeleteVertexArrays(1, &vao_hist);
	vao_hist = 0;

	//cleanup nbody vao
	glBindVertexArray(vao_nbody);
	glDeleteBuffers(1, &vao_nbody_vertices);
	vao_nbody_vertices = 0;
	glDeleteBuffers(1, &vao_nbody_instance_ids);
	vao_nbody_instance_ids = 0;
	glDeleteBuffers(1, &tbo_nbody);
	tbo_nbody = 0;
	glDeleteTextures(1, &tex_nbody);
	tex_nbody = 0;
	if (__M == CUDA){
		cudaGraphicsUnregisterResource(cuda_nbody_vbo_resource);
	}
	glDeleteVertexArrays(1, &vao_nbody);
	vao_nbody = 0;

	checkGLError();
}

void initGL()
{
	int argc = 1;
	const char * argv[] = { "Com4521 Assignment - NBody Visualiser" };

	//glut init
	glutInit(&argc, (char**)argv);  // GLut can't take const char * so unsafe cast

	//init window
	glutInitDisplayMode(GLUT_RGB);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(*argv);

	// glew init (must be done after window creation for some odd reason)
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 "))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		exit(0);
	}

	// register default callbacks
	glutDisplayFunc(displayLoop);
	glutKeyboardFunc(handleKeyboardDefault);
	glutMotionFunc(handleMouseMotionDefault);
	glutMouseFunc(handleMouseDefault);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);


	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)WINDOW_WIDTH / (GLfloat)WINDOW_HEIGHT, 0.001, 10.0);
}

void render(void)
{
	// set view matrix and prepare for rending
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//transformations
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_z, 0.0, 0.0, 1.0);


	//render the densisty field
	if (display_denisty){
		// attach the shader program to rendering pipeline to perform per vertex instance manipulation 
		glUseProgram(vs_hist_program);

		// Bind our Vertex Array Object  (contains vertex buffers object and vertex attribute array)
		glBindVertexArray(vao_hist);

		// Bind and activate texture with instance data (held with the TBO)
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_BUFFER_EXT, tex_hist);

		// Draw the vertices with attached vertex attribute pointers
		glDrawArrays(GL_QUADS, 0, 4 * __D*__D);

		//unbind the vertex array object
		glBindVertexArray(0);

		// Disable the shader program and return to the fixed function pipeline
		glUseProgram(0);
	}

	//render the n bodies
	if (display_bodies){
		// attach the shader program to rendering pipeline to perform per vertex instance manipulation 
		glUseProgram(vs_nbody_program);

		// Bind our Vertex Array Object  (contains vertex buffers object and vertex attribute array)
		glBindVertexArray(vao_nbody);

		// Bind and activate texture with instance data (held with the TBO)
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_BUFFER_EXT, tex_nbody);

		// Draw the vertices with attached vertex attribute pointers
		glDrawArrays(GL_POINTS, 0, 1 * __N);

		//unbind the vertex array object
		glBindVertexArray(0);

		// Disable the shader program and return to the fixed function pipeline
		glUseProgram(0);
	}

	glutSwapBuffers();
	glutPostRedisplay();
}

void checkGLError(){
	int Error;
	if ((Error = glGetError()) != GL_NO_ERROR)
	{
		const char* Message = (const char*)gluErrorString(Error);
		fprintf(stderr, "OpenGL Error : %s\n", Message);
	}
}

void handleKeyboardDefault(unsigned char key, int x, int y)
{
	switch (key) {
		case(27): case('q') : //escape key or q key
			//return control to the users program to allow them to clean-up any allcoated memory etc.
			glutLeaveMainLoop();
			break;
		
		case('b') : //b key
			display_bodies = !display_bodies;
			break;

		case('d') : //d key
			display_denisty = !display_denisty;
			break;
	}
}

void handleMouseDefault(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void handleMouseMotionDefault(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_z += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

#endif

void freeVisualisationMemory() {
    if (__M == CUDA) {
        if (dDensities && Densities) {
            free(const_cast<float*>(Densities));
        }
        if (dBodies && Bodies) {
            free(const_cast<nbody*>(Bodies));
        }
        if (dPositionsX && PositionsX) {
            free(const_cast<float*>(PositionsX));
        }
        if (dPositionsY && PositionsY) {
            free(const_cast<float*>(PositionsY));
        }
    }
    Densities = nullptr;
    dDensities = nullptr;
    Bodies = nullptr;
    dBodies = nullptr;
    PositionsX = nullptr;
    dPositionsX = nullptr;
    PositionsY = nullptr;
    dPositionsY = nullptr;
}