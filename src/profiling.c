#include "profiling.h"

#ifdef ENABLE_PROFILING

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

const char *PROF_STAGE_NAMES[PROF_COUNT] = {
    "Video Decode (ffmpeg open)",
    "First Frame Read",
    "Calibration Extraction",
    "Header Parsing",
    "Frame Acquisition (ffmpeg)",
    "Bit Extraction (tile decode)",
    "RS Decode",
    "File Write (disk I/O)",
    "SHA256 Verification",
    "File Read (disk I/O)",
    "RS Encode",
    "Frame Generation (tile expand)",
    "Video Write (ffmpeg pipe)",
    "Header Build",
};

void prof_report_build(ProfCtx *ctx, ProfReport *report) {
    memset(report, 0, sizeof(ProfReport));
    if (!ctx->active) return;

    int64_t wall_elapsed_ns = prof_clock_ns() - ctx->wall_start_ns;
    report->wall_sec = (double)wall_elapsed_ns / 1.0e9;
    snprintf(report->pipeline_name, sizeof(report->pipeline_name), "%s", ctx->pipeline_name);

    double total_accounted_ns = 0.0;
    report->line_count = 0;

    for (int i = 0; i < PROF_COUNT; ++i) {
        const ProfAccum *a = &ctx->stages[i];
        int64_t main_total = a->total_ns;
        int64_t main_calls = a->call_count;

        int64_t worker_total = prof_worker_total(ctx, (ProfStageId)i);
        int64_t worker_calls = prof_worker_calls(ctx, (ProfStageId)i);

        int64_t total_ns = main_total + worker_total;
        int64_t calls    = main_calls + worker_calls;
        if (calls == 0) continue;

        ProfReportLine *line = &report->lines[report->line_count++];
        line->id         = (ProfStageId)i;
        line->total_ms   = (double)total_ns / 1.0e6;
        line->avg_ms     = (double)total_ns / (double)calls / 1.0e6;
        line->min_ms     = (main_calls > 0) ? (double)a->min_ns / 1.0e6 : 0.0;
        line->max_ms     = (main_calls > 0) ? (double)a->max_ns / 1.0e6 : 0.0;
        line->call_count = calls;
        total_accounted_ns += (double)total_ns;
    }

    report->accounted_ms = total_accounted_ns / 1.0e6;

    if (report->wall_sec > 0.0) {
        double wall_ms = report->wall_sec * 1000.0;
        for (int i = 0; i < report->line_count; ++i)
            report->lines[i].pct_of_total = (report->lines[i].total_ms / wall_ms) * 100.0;
    }
}

void prof_report_print(const ProfReport *report) {
    if (!report || report->line_count == 0) return;

    char name_buf[64];
    snprintf(name_buf, sizeof(name_buf), "%s", report->pipeline_name);
    if (name_buf[0] == '\0') snprintf(name_buf, sizeof(name_buf), "Pipeline");

    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║              VIDCRYPT PROFILING REPORT                          ║\n");
    printf("  ║              Pipeline: %s                         ║\n", name_buf);
    printf("  ╠══════════════════════════════════════════════════════════════════╣\n");
    printf("  ║  Stage                            Time(ms)    %%total    Calls  ║\n");
    printf("  ║  ─────────────────────────────────────────────────────────────── ║\n");

    for (int i = 0; i < report->line_count; ++i) {
        const ProfReportLine *line = &report->lines[i];
        const char *stage_name = PROF_STAGE_NAMES[line->id];
        if (!stage_name) stage_name = "Unknown";

        printf("  ║  %-35s %9.2f   %6.1f%%  %6lld  ║\n",
               stage_name, line->total_ms, line->pct_of_total, (long long)line->call_count);

        if (line->call_count > 1)
            printf("  ║    ─ avg:%.2f  min:%.2f  max:%.2f                    ║\n",
                   line->avg_ms, line->min_ms, line->max_ms);
    }

    printf("  ║  ─────────────────────────────────────────────────────────────── ║\n");
    printf("  ║  %-35s %9.2f   %6.1f%%           ║\n", "Wall Clock Total",
           report->wall_sec * 1000.0, 100.0);
    printf("  ║  %-35s %9.2f   (parallel sum)    ║\n", "Accounted Time (all stages)",
           report->accounted_ms);
    printf("  ║                                                                  ║\n");
    printf("  ║  Note: Accounted time may exceed wall time for parallel stages   ║\n");
    printf("  ║  (e.g., thread pool worker + main thread overlap).               ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

int prof_report_save_csv(const ProfReport *report, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "stage,total_ms,pct_of_total,avg_ms,min_ms,max_ms,calls\n");
    for (int i = 0; i < report->line_count; ++i) {
        const ProfReportLine *line = &report->lines[i];
        const char *stage_name = PROF_STAGE_NAMES[line->id];
        if (!stage_name) stage_name = "Unknown";
        fprintf(f, "%s,%.3f,%.1f,%.3f,%.3f,%.3f,%lld\n",
                stage_name, line->total_ms, line->pct_of_total,
                line->avg_ms, line->min_ms, line->max_ms, (long long)line->call_count);
    }
    fclose(f);
    return 0;
}

#endif /* ENABLE_PROFILING */
