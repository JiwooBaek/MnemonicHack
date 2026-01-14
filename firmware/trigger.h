#pragma once
/*
 * trigger.h — Header-only trigger utilities for STM32 (libopencm3) + host stub
 *
 * Features:
 * - Configurable GPIO port/pin settings via common macros.
 * - Thread-safe initialization: trigger_init_once() (idempotent).
 * - Basic control: trigger_start() (High) / trigger_end() (Low).
 * - Busy-wait delay: sleep_us() / sleep_ms() (calibratable).
 * - Pulse generation: trig_pulse_us() / trig_pulse_ms().
 * - Frame marking: trigger_frame_mark_begin() / end() (long pulse for frame delimiters).
 *
 * Build Flags:
 * -DTRIG_GPIO_PORT=GPIOD -DTRIG_GPIO_PIN=GPIO2 -DTRIG_GPIO_RCC=RCC_GPIOD
 * -DTRIG_FRAME_MARK_MS=5
 * -DTRIG_US_LOOP_PER_US=30    // Adjust based on CPU clock (loop count per microsecond)
 * -DTRIGGER_DISABLE           // Stubs all functions (useful for host-side logging)
 */

#ifdef __cplusplus
extern "C" {
#endif

/* ===== User Configuration (Override via -D compiler flag) ===== */
#ifndef TRIG_GPIO_PORT
#define TRIG_GPIO_PORT GPIOD
#endif
#ifndef TRIG_GPIO_PIN
#define TRIG_GPIO_PIN  GPIO2
#endif
#ifndef TRIG_GPIO_RCC
#define TRIG_GPIO_RCC  RCC_GPIOD
#endif

/* Default duration (ms) for frame markers (start/end of sequence) */
#ifndef TRIG_FRAME_MARK_MS
#define TRIG_FRAME_MARK_MS 5U
#endif

/* Calibration value for busy-wait loop (iterations per microsecond) */
#ifndef TRIG_US_LOOP_PER_US
#define TRIG_US_LOOP_PER_US 30U
#endif
/* ============================================================= */

#include <stdint.h>
#include <stdbool.h>

/* ===== Disabled Mode (Stub Implementation) ===== */
#ifdef TRIGGER_DISABLE
static inline void trigger_init_once(void) {}
static inline void trigger_start(void) {}
static inline void trigger_end(void) {}
static inline void sleep_us(uint32_t us) { (void)us; }
static inline void sleep_ms(uint32_t ms) { (void)ms; }
static inline void trig_pulse_us(uint32_t us) { (void)us; }
static inline void trig_pulse_ms(uint32_t ms) { (void)ms; }
static inline void trigger_frame_mark_begin(void) {}
static inline void trigger_frame_mark_end(void) {}
#else

/* ===== Target Build: STM32 (libopencm3) ===== */
#if defined(__arm__) || defined(__ARMEL__)
#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/gpio.h>

static inline void sleep_us(uint32_t us) {
  for (volatile uint32_t i = 0; i < (us * (uint32_t)TRIG_US_LOOP_PER_US); ++i) {
    __asm__ volatile("nop");
  }
}
static inline void sleep_ms(uint32_t ms) {
  while (ms--) sleep_us(1000U);
}

static inline void trigger_init_once(void) {
  static bool inited = false;
  if (inited) return;

  rcc_periph_clock_enable(TRIG_GPIO_RCC);

  /* Support for both new and legacy libopencm3 APIs */
#if defined(GPIO_MODE_OUTPUT) && defined(GPIO_PUPD_NONE)
  /* F2/F3/F4 and newer families */
  gpio_mode_setup(TRIG_GPIO_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, TRIG_GPIO_PIN);
#else
  /* F1 family legacy macros */
  gpio_set_mode(TRIG_GPIO_PORT, GPIO_MODE_OUTPUT_2_MHZ, GPIO_CNF_OUTPUT_PUSHPULL, TRIG_GPIO_PIN);
#endif

  /* Set default state to Low */
  gpio_clear(TRIG_GPIO_PORT, TRIG_GPIO_PIN);
  inited = true;
}

static inline void trigger_start(void) { gpio_set(TRIG_GPIO_PORT, TRIG_GPIO_PIN); }
static inline void trigger_end(void)   { gpio_clear(TRIG_GPIO_PORT, TRIG_GPIO_PIN); }

/* ===== Host Build: Simulation and Unit Testing ===== */
#else
static inline void sleep_us(uint32_t us) { (void)us; }
static inline void sleep_ms(uint32_t ms) { (void)ms; }

static inline void trigger_init_once(void) {}
static inline void trigger_start(void) {}
static inline void trigger_end(void) {}
#endif /* platform select */

/* ===== Common Utility Functions ===== */
static inline void trig_pulse_us(uint32_t us) {
  trigger_start();
  sleep_us(us);
  trigger_end();
}

static inline void trig_pulse_ms(uint32_t ms) {
  trigger_start();
  sleep_ms(ms);
  trigger_end();
}

/* "Frame Marker" (long pulse) to indicate the beginning/end of a waveform sequence */
static inline void trigger_frame_mark_begin(void) { trig_pulse_ms(TRIG_FRAME_MARK_MS); }
static inline void trigger_frame_mark_end(void)   { trig_pulse_ms(TRIG_FRAME_MARK_MS); }

#endif /* TRIGGER_DISABLE */

#ifdef __cplusplus
}
#endif