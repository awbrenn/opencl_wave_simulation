#define DARK_WATER (float4)(0.05f, 0.05f, 0.2f, 1.0f)
#define LIGHT_WATER (float4)(0.3f,0.7f,0.8f,1.0f)
#define SKY_COLOR (float4)(1.0f,1.0f,0.9f,1.0f)
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
            new_energy += omega[DIRECTIONS*k+n]*from[store(ry,rx,n)];
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
        height += (energy.s0+energy.s1+energy.s2+energy.s3+energy.s4+energy.s5+energy.s6+energy.s7)/SCALE;
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

    // get the y values of adjacent vertices
    if(rx==0) { y2=0.0f; }
    else if(rx==(WIDTH-1)){ y1=1.0f; }
    else
    { 
        y2=rbuffer[(rx-1)+ry*WIDTH].y;
        y1=rbuffer[(rx+1)+ry*WIDTH].y;
    }

    if(ry==0) { y4=0.0f; }
    else if(ry==(LENGTH-1)) { y3=1.0f; }
    else
    {
        y4=rbuffer[rx+(ry-1)*WIDTH].y;
        y3=rbuffer[rx+(ry+1)*WIDTH].y;
    }

    // calculate components of normal vector
    normal=(float4) ((y2-y1), (2*((float)SCALE))/((float)WIDTH),
           (((float)LENGTH)/((float)WIDTH))*(y4-y1), 1.0f);
    
    //make unit length
    normal/=sqrt(normal.x*normal.x+normal.y*normal.y+normal.z*normal.z);

    nbuffer[i] = normal;

    // nbuffer[i]=(float4)(0.0f,1.0f,0.0f,1.0f);
}

__kernel void colors(__global float4* rbuffer, float4 lightdir,
    float4 viewdir, __global float4* nbuffer)
{
    int rx = get_global_id(0);
    int ry = get_global_id(1);
    int color_index = rx+ry*WIDTH + COLOR_OFFSET;
    int normal_index = rx+ry*WIDTH;
    float4 dark_water = DARK_WATER;
    float4 light_water = LIGHT_WATER;
    float4 sky_color = SKY_COLOR;
    float4 normal = nbuffer[normal_index];
    float fresnel = FRESNEL;
    float sea_view = (-1.0f)*min(0.0f, (float)(dot(normal,viewdir)));

    float4 color = (dot(lightdir,normal))*(mix(dark_water,light_water,sea_view))+
                    fresnel*sky_color*(pow((1.0f)-sea_view,5.0f));

    rbuffer[color_index] = color;
}