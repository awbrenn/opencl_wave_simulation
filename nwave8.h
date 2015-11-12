//wave.h

#define WIDTH 768
#define LENGTH 768
#define LWS 8
#define SCALE (4.0f)
#define DIRECTIONS 5
#define SIZE (WIDTH*LENGTH*DIRECTIONS)
#define COLOR_OFFSET (WIDTH*LENGTH)

// Here we reorder storage from the natural, [LENGTH][WIDTH][DIRECTIONS],
// to one that works better with memory accesses, [LENGTH][DIRECTIONS][WIDTH].
#define store(i,j,k) ((i)*(WIDTH*DIRECTIONS)+(k)*WIDTH+(j))
#define dist(i,j,k) dist[store(i,j,k)]
