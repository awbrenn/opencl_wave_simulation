#define store(row,col) (DIRECTIONS*(row)+(col))

__kernel void update(__global double* from, __global double* to, 
    __global int* dist, __constant double* omega)
{
int k, n;
double new_energy;
int j = get_global_id(0);

for(k=0;k<DIRECTIONS;k++){
    new_energy = from[store(j,k)];
    for(n=0;n<DIRECTIONS;n++){
        new_energy += omega[store(k,n)]*from[store(j,n)];
        }
    to[store(dist[store(j,k)],k)] = new_energy;
    }
}

__kernel void heights(__global float4* rbuffer, __global float4* f)
{
    // stubbed
}

__kernel void normals(__global float4* rbuffer, __global float4* nbuffer)
{
    // stubbed
}

__kernel void colors(__global float4* rbuffer, __global float4* lightdir,
    __global float4* viewdir, __global float4* nbuffer)
{
    // stubbed
}