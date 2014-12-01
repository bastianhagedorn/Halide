#ifndef HALIDE_HALIDERUNTIMEOPENGL_H
#define HALIDE_HALIDERUNTIMEOPENGL_H

#include "HalideRuntime.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \file
 *  Routines specific to the Halide OpenGL runtime.
 */

extern WEAK struct halide_device_interface *halide_glsl;

/** These are forward declared here to allow clients to override the
 *  Halide Glsl runtime. Do not call them. */
// @{
extern WEAK int halide_opengl_initialize_kernels(void *user_context, void **state_ptr,
                                                 const char *src, int size);

extern WEAK int halide_opengl_run(void *user_context,
                                  void *state_ptr,
                                  const char *entry_name,
                                  int blocksX, int blocksY, int blocksZ,
                                  int threadsX, int threadsY, int threadsZ,
                                  int shared_mem_bytes,
                                  size_t arg_sizes[],
                                  void *args[],
                                  int8_t is_buffer[]);
// @}

/** This function is called to populate the buffer_t.dev field with a constant
 * indicating that the OpenGL object corresponding to the buffer_t is bound by
 * the app and not by the Halide runtime. For example, the buffer_t may be
 * backed by an FBO already bound by the application. */
extern uint64_t halide_opengl_output_client_bound();

/** Forget all state associated with the previous OpenGL context.  This is
 * similar to halide_opengl_release, except that we assume that all OpenGL
 * resources have already been reclaimed by the OS. */
extern void halide_opengl_context_lost(void *user_context);

/** This functions MUST be provided by the host environment to retrieve pointers
 *  to OpenGL API functions. */
void *halide_opengl_get_proc_address(void *user_context, const char *name);


/** This functions MUST be provided by the host environment to create an OpenGL
 *  context for use by the OpenGL backend. */
int halide_opengl_create_context(void *user_context);

#ifdef __cplusplus
} // End extern "C"
#endif

#endif // HALIDE_HALIDERUNTIMEOPENGL_H