// #define store(row,col) (DIRECTIONS*(row)+(col))

__kernel void update(__global float8* from, __global float8* to, 
    __global int* dist, __constant float8* omega)
{
    // int k, n;
    // float8 new_energy;
    // int rx = get_global_id(0);
    // int ry = get_global_id(1);

    // for(k=0;k<DIRECTIONS;k++){
    //     new_energy = from[store(ry,rx,k)];
    //     for(n=0;n<DIRECTIONS;n++){
    //         new_energy += omega[store(ry,rx,n)]*from[store(ry,rx,n)];
    //         }
    //     to[store(dist[store(ry,rx,k)],rx,k)] = new_energy;
    //     }
}

__kernel void heights(__global float4* rbuffer, __global float8* f)
{
    int k;
    float8 energy;
    float height=0.0f;
    int rx = get_global_id(0);
    int ry = get_global_id(1);
    int index = rx+ry*WIDTH;
    // for(k=0;k<DIRECTIONS;k++){
    //     energy = f[store(ry,rx,k)];
    //     height += energy.s0+energy.s1+energy.s2+energy.s3+energy.s4+energy.s5+energy.s6+energy.s7;
    // }
    rbuffer[index].x = SCALE*((float)rx/(float)(WIDTH-1))-(SCALE/2.0f);
    rbuffer[index].y = height;
    rbuffer[index].z = SCALE*((float)ry/(float)(LENGTH-1))-(SCALE/2.0f);
    rbuffer[index].w = 1.0f;
}

__kernel void normals(__global float4* rbuffer, __global float4* nbuffer)
{
    int rx = get_global_id(0);
    int ry = get_global_id(1);
    int index = rx+ry*WIDTH;

    nbuffer[index].x = 0.0f;
    nbuffer[index].y = 1.0f;
    nbuffer[index].z = 0.0f;
    nbuffer[index].w = 1.0f;
}

__kernel void colors(__global float4* rbuffer, float4 lightdir,
    float4 viewdir, __global float4* nbuffer)
{
    int rx = get_global_id(0);
    int ry = get_global_id(1);
    int i = rx+ry*WIDTH + COLOR_OFFSET;

    rbuffer[i] = (float4)(0.2f, 0.7f, 1.0f, 1.0f);
}