//
// Wave model with 8 wave numbers and an obstruction.  Lighting the obstruction
// is on the CPU; lighting the waves is on the GPU.
//
// This version uses indexed VBOs and just draws QUADS.

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glx.h>
#include <unistd.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_platform.h>
#include <CL/opencl.h>
#include "RGU.h"

#include "nwave8.h"

#define BLOCKLIST 1
#define FINALTIME 10000
#define RENDER_STEPS 4
#define WAVENUMBERS 8 

struct ivector {
	int x;
	int y;
	};

struct ivector ci[] = {
	0,0, 1,0, -1,0, 0,1, 0,-1
	};

int reverse[] = {
	0, 2, 1, 4, 3
	};

int nclass[LENGTH][WIDTH];

#define DATA_SIZE (WIDTH*LENGTH*4*sizeof(float))
#define INDEX_SIZE ((WIDTH-1)*(LENGTH-1)*4*sizeof(unsigned int))

#define VERTICES 0
#define INDICES 1

GLuint bufs[2];
unsigned int indices[(WIDTH-1)*(LENGTH-1)][4];

// Since OpenCL takes 1D arrays, we might as well switch to those.
// This does have the advantage that the dist[] pointer can be stored
// as a single index.  
float omega[DIRECTIONS*DIRECTIONS][WAVENUMBERS];
float f[2][SIZE][WAVENUMBERS];
int dist[SIZE];			// where to send the flow

// Usual game to improve readability
#define f(h,i,j,k,m) f[h][store(i,j,k)][m]
#define omega(i,j,k) omega[(i)*DIRECTIONS+j][k]

#define INSIDE 0
#define BORDER 1

int iwcx(double wcx)
{
return ((int)trunc(((double)(WIDTH))*(wcx/(double)(SCALE) + 0.5)));
}

int iwcz(double wcz)
{
return ((int)trunc(((double)(LENGTH))*(wcz/(double)(SCALE) + 0.5)));
}

void geometry()
{
int i,j,k,ty,tx;

for(i=0;i<LENGTH;i++){
	for(j=0;j<WIDTH;j++){
		nclass[i][j] = INSIDE;
		}
	}
// Put in an obstructing box (base).  This has to be in terms of
// WIDTH and LENGTH grid points, and so we have to invert wc. 
for(i=iwcz(SCALE/8.0); i<iwcz(SCALE/4.0); i++){
	for(j=iwcx(-SCALE/4.0);j<iwcx(-SCALE/8.0);j++){
		nclass[i][j] = BORDER;
		}
	}

// For node (i,j), direction k flow out goes to
// node (i+ci[k].y,j+ci[k].x), also in direction k.
for(i=0;i<LENGTH;i++){
	for(j=0;j<WIDTH;j++){
		for(k=0;k<DIRECTIONS;k++){
			// Mod needs help with -1.
			ty = (i+ci[k].y + LENGTH)%LENGTH;	
			tx = (j+ci[k].x + WIDTH)%WIDTH;	
			if(nclass[ty][tx]==INSIDE){ 
				dist(i,j,k) = store(ty,tx,k);
				}
			else { // Flow would go into BORDER; reverse.
				dist(i,j,k) = store(i,j,reverse[k]);
				}
			}
		}
	}
}

#define PEAK (4.0)

void init_lattice()
{
// Load densities.
int i,j,k,m;
float base;

for(i=0;i<LENGTH;i++){
	for(j=0;j<WIDTH;j++){
		if(nclass[i][j]==BORDER){
			for(k=0;k<DIRECTIONS;k++){
				for(m=0;m<WAVENUMBERS;m++){
					f(0,i,j,k,m) = 0.0;
					f(1,i,j,k,m) = 0.0;
					}
				}
			}
		else {
			// These are interior nodes.
			for(k=0;k<DIRECTIONS;k++){
				for(m=0;m<WAVENUMBERS;m++){
					f(0,i,j,k,m) = 0.0;
					f(1,i,j,k,m) = 0.0;
					}
				}
			// Start a left-to-right wave; max height is PEAK.
			base = PEAK;
			base *= 1.0 - fabs((float)(i)-0.7*LENGTH)/(0.2*LENGTH);
			base *= 1.0 - fabs((float)(j)-0.08*WIDTH)/(0.01*WIDTH); 
			if(j>0.07*WIDTH && j<0.09*WIDTH && i>0.5*LENGTH && 
				i<0.9*LENGTH) {
				for(m=0;m<WAVENUMBERS;m++){
					f(0,i,j,1,m)= base/(float)(WAVENUMBERS);
					}
				}
			}
		}
	}
}

void load_omega(float K,int slot)
{
int i,j;

omega(0,0,slot) = -4.0*K;
for(j=1;j<DIRECTIONS;j++) omega(0,j,slot) = 2.0-4.0*K;
for(i=1;i<DIRECTIONS;i++) omega(i,0,slot) = K;
for(i=1;i<3;i++){
	for(j=1;j<3;j++) omega(i,j,slot) = K-1.0;
	for(j=3;j<5;j++) omega(i,j,slot) = K;
	}
for(i=3;i<5;i++){
	for(j=1;j<3;j++) omega(i,j,slot) = K;
	for(j=3;j<5;j++) omega(i,j,slot) = K-1.0;
	}
}

cl_platform_id myplatform;
cl_context mycontext;
cl_device_id *mydevice;
cl_command_queue mycq;
cl_kernel mykrn_update, mykrn_heights, mykrn_normals, mykrn_colors;
cl_program myprogram;
cl_mem focl[2], dist_ocl, omega_ocl, rbuffer_ocl, nbuffer_ocl; 
cl_int err;
char* oclsource; 
size_t ws[2] = {WIDTH,LENGTH}; 
size_t lws[2] = {LWS,LWS}; 


void render()
{
glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
// Light only the obstructions; wave lighting is in the colors kernel.
glEnable(GL_LIGHTING);
glCallList(BLOCKLIST);
glDisable(GL_LIGHTING);
glDrawElements(GL_QUADS,(WIDTH-1)*(LENGTH-1)*4,GL_UNSIGNED_INT,NULL);
glutSwapBuffers();
}

void run_updates()
{
static int from = 0;
static int t = 0;
cl_event wait[1];

clSetKernelArg(mykrn_update,0,sizeof(cl_mem),(void *)&focl[from]);
clSetKernelArg(mykrn_update,1,sizeof(cl_mem),(void *)&focl[1-from]);
clEnqueueNDRangeKernel(mycq,mykrn_update,2,NULL,ws,lws,0,0,&wait[0]);
clWaitForEvents(1,wait);
if(t%RENDER_STEPS==0) {
	glFinish();
	clEnqueueAcquireGLObjects(mycq,1,&rbuffer_ocl,0,0,0);
	clSetKernelArg(mykrn_heights,1,sizeof(cl_mem),(void *)&focl[1-from]);
	clEnqueueNDRangeKernel(mycq,mykrn_heights,2,NULL,ws,lws,0,0,&wait[0]);
	clWaitForEvents(1,wait);
	clEnqueueNDRangeKernel(mycq,mykrn_normals,2,NULL,ws,lws,0,0,&wait[0]);
	clWaitForEvents(1,wait);
	clEnqueueNDRangeKernel(mycq,mykrn_colors,2,NULL,ws,lws,0,0,&wait[0]);
	clWaitForEvents(1,wait);
	clEnqueueReleaseGLObjects(mycq,1,&rbuffer_ocl,0,0,0);
	clFinish(mycq);
	glutPostRedisplay();
	}
from = 1-from;
t++;
usleep(10000);
}

float lightdir[4];
float viewdir[4];

void buffers()
{
focl[0] = clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	SIZE*sizeof(cl_float8),&f[0][0][0],&err);
focl[1] = clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	SIZE*sizeof(cl_float8),&f[1][0][0],&err);
dist_ocl = clCreateBuffer(mycontext,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
	SIZE*sizeof(int),&dist[0],&err);
omega_ocl = clCreateBuffer(mycontext,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
	DIRECTIONS*DIRECTIONS*sizeof(cl_float8),&omega[0][0],&err);
rbuffer_ocl = clCreateFromGLBuffer(mycontext,CL_MEM_READ_WRITE,bufs[VERTICES],
	&err);
nbuffer_ocl = clCreateBuffer(mycontext,CL_MEM_READ_WRITE,WIDTH*LENGTH*
	sizeof(cl_float4),NULL,&err);

clSetKernelArg(mykrn_update,2,sizeof(cl_mem),(void *)&dist_ocl);
clSetKernelArg(mykrn_update,3,sizeof(cl_mem),(void *)&omega_ocl);
clSetKernelArg(mykrn_heights,0,sizeof(cl_mem), (void *)&rbuffer_ocl);
clSetKernelArg(mykrn_normals,0,sizeof(cl_mem), (void *)&rbuffer_ocl);
clSetKernelArg(mykrn_normals,1,sizeof(cl_mem), (void *)&nbuffer_ocl);
clSetKernelArg(mykrn_colors,0,sizeof(cl_mem), (void *)&rbuffer_ocl);
clSetKernelArg(mykrn_colors,1,sizeof(cl_float4), &lightdir);
clSetKernelArg(mykrn_colors,2,sizeof(cl_float4), (void *)&viewdir);
clSetKernelArg(mykrn_colors,3,sizeof(cl_mem), (void *)&nbuffer_ocl);
}

void initCL()
{
size_t program_length;
const char *header;
int err;
unsigned int gpudevcount;

err = RGUGetPlatformID(&myplatform);

err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,0,NULL,&gpudevcount);
fprintf(stderr,"device count %d\n",gpudevcount);

mydevice = new cl_device_id[gpudevcount];
err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,gpudevcount,mydevice,NULL);

cl_context_properties props[] = {
	CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
	CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
	CL_CONTEXT_PLATFORM, (cl_context_properties)myplatform, 
	0};
mycontext = clCreateContext(props,1,&mydevice[0],NULL,NULL,&err);
fprintf(stderr,"context creation %d\n",err);

mycq = clCreateCommandQueue(mycontext,mydevice[0],0,&err);
fprintf(stderr,"command queue %d\n",err);

header = RGULoadProgSource("nwave8.h","",&program_length);
oclsource = RGULoadProgSource("nwave8.cl",header,&program_length);

myprogram = clCreateProgramWithSource(mycontext,1,
	(const char **)&oclsource, &program_length, &err);
fprintf(stderr,"create program %d\n",err);

clBuildProgram(myprogram, 0, NULL, NULL, NULL, NULL);

mykrn_update = clCreateKernel(myprogram, "update", &err);
if(err==CL_SUCCESS) fprintf(stderr,"kernel build update ok\n");
else fprintf(stderr,"kernel build err update %d\n",err);
mykrn_heights = clCreateKernel(myprogram, "heights", &err);
if(err==CL_SUCCESS) fprintf(stderr,"kernel build heights ok\n");
else fprintf(stderr,"kernel build err heights %d\n",err);
mykrn_normals = clCreateKernel(myprogram, "normals", &err);
if(err==CL_SUCCESS) fprintf(stderr,"kernel build normals ok\n");
else fprintf(stderr,"kernel build err normals %d\n",err);
mykrn_colors = clCreateKernel(myprogram, "colors", &err);
if(err==CL_SUCCESS) fprintf(stderr,"kernel build colors ok\n");
else fprintf(stderr,"kernel build err colors %d\n",err);
}

float dot(float a[4], float b[4])
{
float result;
result = a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
return(result);
}

void setup_the_viewvol()
{
float eyep[3], view[3], up[3];
float length;
int i;
glEnable(GL_DEPTH_TEST);
glEnable(GL_MULTISAMPLE_ARB);
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
gluPerspective(40.0,1.0,0.1,40.0);
glMatrixMode(GL_MODELVIEW);
glLoadIdentity();

eyep[0] = 4.0; eyep[1] = 2.0; eyep[2] = 3.0;
view[0] = 0.0; view[1] = 0.0; view[2] = 0.0;
up[0] = 0.0; up[1] = 1.0; up[2] = 0.0;

for(i=0;i<3;i++) viewdir[i] = view[i]-eyep[i];
length = sqrt(dot(viewdir,viewdir));
for(i=0;i<3;i++) viewdir[i] /= length;
viewdir[3] = 0.0;

gluLookAt(eyep[0],eyep[1],eyep[2],view[0],view[1],view[2],up[0],up[1],up[2]);
}


void do_lights()
{
float light0_ambient[] = { 0.0, 0.0, 0.0, 1.0 };
float light0_diffuse[] = { 2.0, 2.0, 2.0, 1.0 };
float light0_specular[] = { 1.0, 1.0, 1.0, 1.0 };
float light0_position[] = {3.0, 3.0, 3.0, 1.0 };
float light0_direction[] = {-3.0,-3.0,-3.0,1.0};
float length;
int i;

// We need the light (sun) direction for the lighting kernel.
length = sqrt(dot(light0_direction,light0_direction));
for(i=0;i<3;i++) lightdir[i] = -light0_direction[i]/length;
lightdir[3] = 0.0;

glLightModelfv(GL_LIGHT_MODEL_AMBIENT,light0_ambient);
glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,1);

glLightfv(GL_LIGHT0,GL_AMBIENT,light0_ambient);
glLightfv(GL_LIGHT0,GL_DIFFUSE,light0_diffuse);
glLightfv(GL_LIGHT0,GL_SPECULAR,light0_specular);
glLightf(GL_LIGHT0,GL_SPOT_EXPONENT,1.0);
glLightf(GL_LIGHT0,GL_SPOT_CUTOFF,180.0);
glLightf(GL_LIGHT0,GL_CONSTANT_ATTENUATION,0.9);
glLightf(GL_LIGHT0,GL_LINEAR_ATTENUATION,0.3);
glLightf(GL_LIGHT0,GL_QUADRATIC_ATTENUATION,0.01);
glLightfv(GL_LIGHT0,GL_POSITION,light0_position);
glLightfv(GL_LIGHT0,GL_SPOT_DIRECTION,light0_direction);
glEnable(GL_LIGHT0);
}

void do_material_block()
{
float mat_ambient[] = {0.0,0.0,0.0,0.0};
float mat_diffuse[] = {0.9,0.7,0.3,1.0};
float mat_specular[] = {1.0,1.0,1.0,1.0};
float mat_shininess[] = {2.0};

glMaterialfv(GL_FRONT,GL_AMBIENT,mat_ambient);
glMaterialfv(GL_FRONT,GL_DIFFUSE,mat_diffuse);
glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular);
glMaterialfv(GL_FRONT,GL_SHININESS,mat_shininess);
}

#define QSCALE (0.25*(float)(SCALE))
#define ESCALE (0.125*(float)(SCALE))
#define BLOCKH (0.30*(float)(SCALE))

void block_geometry()
{
int i;
float front[4][3] = {{-QSCALE,0.0,QSCALE},{-ESCALE,0.0,QSCALE},
		{-ESCALE,BLOCKH,QSCALE},{-QSCALE,BLOCKH,QSCALE}};
float right[4][3] = {{-ESCALE,0.0,QSCALE},{-ESCALE,0.0,ESCALE},
		{-ESCALE,BLOCKH,ESCALE},{-ESCALE,BLOCKH,QSCALE}};
float top[4][3] = {{-ESCALE,BLOCKH,QSCALE},{-ESCALE,BLOCKH,ESCALE},
		{-QSCALE,BLOCKH,ESCALE},{-QSCALE,BLOCKH,QSCALE}};
glBegin(GL_QUADS);
glNormal3f(0.0,0.0,1.0);
for(i=0;i<4;i++) glVertex3fv(front[i]);
glNormal3f(1.0,0.0,0.0);
for(i=0;i<4;i++) glVertex3fv(right[i]);
glNormal3f(0.0,1.0,0.0);
for(i=0;i<4;i++) glVertex3fv(top[i]);
glEnd();
}

void static_load_arrays()
{
init_lattice();
load_omega(0.50,0);
load_omega(0.45,1);
load_omega(0.40,2);
load_omega(0.35,3);
load_omega(0.30,4);
load_omega(0.25,5);
load_omega(0.20,6);
load_omega(0.15,7);
}

void load_indices()
{
int i, j;
for(i=0;i<LENGTH-1;i++){
        for(j=0;j<WIDTH-1;j++){
                indices[i*(WIDTH-1)+j][0] = i*WIDTH+j;
                indices[i*(WIDTH-1)+j][1] = (i+1)*WIDTH+j;
                indices[i*(WIDTH-1)+j][2] = (i+1)*WIDTH+1+j;
                indices[i*(WIDTH-1)+j][3] = i*WIDTH+1+j;
                }
        }
}

void initGL(int argc, char **argv)
{
int i;
glutInit(&argc,argv);
glutInitDisplayMode(GLUT_RGBA|GLUT_DEPTH|GLUT_DOUBLE|GLUT_MULTISAMPLE);
glutInitWindowSize(768,768);
glutInitWindowPosition(100,50);
glutCreateWindow("my_cool_cube");
setup_the_viewvol();
do_lights();
load_indices();
glClearColor(0.45,0.45,0.45,1.0);
glewInit();
glGenBuffers(2,bufs);
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs[INDICES]);
glBufferData(GL_ELEMENT_ARRAY_BUFFER, INDEX_SIZE, indices, GL_STATIC_DRAW);
glBindBuffer(GL_ARRAY_BUFFER, bufs[VERTICES]);
glBufferData(GL_ARRAY_BUFFER, 2*DATA_SIZE, 0, GL_DYNAMIC_DRAW);
glVertexPointer(4,GL_FLOAT,4*sizeof(GLfloat),(GLfloat *)0);
glColorPointer(4,GL_FLOAT,4*sizeof(GLfloat),(GLfloat *)(DATA_SIZE));
glEnableClientState(GL_VERTEX_ARRAY);
glEnableClientState(GL_COLOR_ARRAY);
glNewList(BLOCKLIST,GL_COMPILE);
do_material_block();
block_geometry();
glEndList();
}

void cleanup()
{
clReleaseKernel(mykrn_update); 
clReleaseKernel(mykrn_heights); 
clReleaseKernel(mykrn_normals);
clReleaseKernel(mykrn_colors);
clReleaseProgram(myprogram);
clReleaseCommandQueue(mycq);
glDeleteBuffers(2,bufs);
clReleaseMemObject(focl[0]);
clReleaseMemObject(focl[1]);
clReleaseMemObject(dist_ocl);
clReleaseMemObject(omega_ocl);
clReleaseMemObject(rbuffer_ocl);
clReleaseMemObject(nbuffer_ocl);
clReleaseContext(mycontext);
exit(0);
}

void getout(unsigned char key, int x, int y)
{
switch(key) {
	case 'q': 		
                cleanup();
        	break;
	case 'p': 		
		usleep(4000000);
        	break;
	default:
        	break;
    }
}

int main(int argc, char **argv)
{
geometry();
static_load_arrays();
initGL(argc,argv);
initCL();
buffers();
glutIdleFunc(run_updates);
glutDisplayFunc(render);
glutKeyboardFunc(getout);
glutMainLoop();
return 1;
}
