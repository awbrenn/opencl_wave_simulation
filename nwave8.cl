/*
    Author: Austin Brennan
    Date:   11/19/2015
    Course: CPSC 6780 - Dr. Robert Geist

    Description:
            The update and height kenrels use
            the Lattice Boltzmann method to calcule
            wave heights on a 2D lattice. These
            hieghts are then colored using the colors
            kernel. The colors kernel uses the normals
            calculated in the normals kernel to create
            more wave like shading on the grid.
*/

#define WAVE_SCALE (4.0f)
#define DAMP_FACTOR (0.9f)
#define DARK_WATER ((float4)(0.05f, 0.05f, 0.2f, 1.0f))
#define LIGHT_WATER ((float4)(0.3f,0.7f,0.8f,1.0f))
#define SKY_COLOR ((float4)(1.0f,1.0f,0.9f,1.0f))
#define FRESNEL (0.5f)


__kernel void update(__global float8* from, __global float8* to, 
    __global int* dist, __constant float8* omega)
{
    int k, n;
    float8 new_energy;
    int rx = get_global_id(0);
    int ry = get_global_id(1);

    for(k=0;k<DIRECTIONS;k++){
        new_energy = from[store(ry,rx,k)];
        for(n=0;n<DIRECTIONS;n++){
            new_energy += DAMP_FACTOR*omega[DIRECTIONS*k+n]*from[store(ry,rx,n)];
            }
        to[dist(ry,rx,k)] = new_energy;
        }
}

__kernel void heights(__global float4* rbuffer, __global float8* f)
{
    int k;
    float8 energy;
    float height=0.0f;
    int rx = get_global_id(0);
    int ry = get_global_id(1);

    int index = rx+ry*WIDTH;

    for(k=0;k<DIRECTIONS;k++){
        energy = f[store(ry,rx,k)];
        height += (energy.s0+energy.s1+energy.s2+energy.s3+energy.s4+energy.s5+energy.s6+energy.s7)/WAVE_SCALE;
    }
    rbuffer[index].x = SCALE*((float)rx/(float)(WIDTH-1))-(SCALE/2.0f);
    rbuffer[index].y = height;
    rbuffer[index].z = SCALE*((float)ry/(float)(LENGTH-1))-(SCALE/2.0f);
    rbuffer[index].w = 1.0f;
}

__kernel void normals(__global float4* rbuffer, __global float4* nbuffer)
{
    int rx = get_global_id(0);
    int ry = get_global_id(1);
    int i = rx+ry*WIDTH;
    float y1,y2,y3,y4;
    float4 normal;

    // set boarder vertices y values of neighbors to zero
    if(rx==0) {
        y2=0.0f;
        y1=rbuffer[(rx+1)+ry*WIDTH].y;
    }
    else if(rx==(WIDTH-1)){
        y1=0.0f;
        y2=rbuffer[(rx-1)+ry*WIDTH].y;
    }
    else {
        y2=rbuffer[(rx-1)+ry*WIDTH].y; 
        y1=rbuffer[(rx+1)+ry*WIDTH].y;
    }

    if(ry==0) {
        y4=0.0f;
        y3=rbuffer[rx+(ry+1)*WIDTH].y;
    }
    else if(ry==(LENGTH-1)) {
        y3=0.0f;
        y4=rbuffer[rx+(ry-1)*WIDTH].y;
    }
    else {
        y4=rbuffer[rx+(ry-1)*WIDTH].y;
        y3=rbuffer[rx+(ry+1)*WIDTH].y;
    }

    // calculate components of normal vector
    normal=(float4) ((y2-y1), (2*((float)SCALE))/((float)WIDTH),
           (((float)LENGTH)/((float)WIDTH))*(y4-y1), 1.0f);
    
    //make normal unit length
    normal/=sqrt(normal.x*normal.x+normal.y*normal.y+normal.z*normal.z);

    nbuffer[i] = normal;
}

__kernel void colors(__global float4* rbuffer, float4 lightdir,
    float4 viewdir, __global float4* nbuffer)
{
    int rx = get_global_id(0);
    int ry = get_global_id(1);
    float4 normal = nbuffer[rx+ry*WIDTH];
    float sea_view = (-1.0f)*min(0.0f, (float)(dot(normal,viewdir)));

    float4 color = (dot(lightdir,normal))*(mix(DARK_WATER,LIGHT_WATER,sea_view))+
                    FRESNEL*SKY_COLOR*(pow((1.0f)-sea_view,5.0f));

    rbuffer[rx+ry*WIDTH + COLOR_OFFSET] = color;
}