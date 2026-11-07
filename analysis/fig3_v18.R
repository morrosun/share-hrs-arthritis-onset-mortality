# Figure 3 (v18): age-specific mortality by arthritis state
# 改动（相对 v5）：数据源改为按 v18 最终队列口径重算的 fig3_rates_v18.csv
#   —— 同年死亡插补 0.5 年、剔除入组 <40 岁者，风险集与 Table 4 完全一致
#   （34,568 人 / 13,389 死亡）。
Sys.setlocale("LC_ALL", "en_US.UTF-8")
setwd(Sys.getenv("AGING_DB_ROOT", "."))
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(scales)})

d <- read.csv("out/fig3_rates_v18.csv", fileEncoding = "UTF-8-BOM")
d$grp <- trimws(as.character(d$grp))

PAL <- c("never"      = "#8A9096",
         "prevalent"  = "#B03A2E",
         "on_<60"     = "#1B3A5C",
         "on_60-74"   = "#4C7EA8",
         "on_>=75"    = "#9DC0D8")

LAB <- c("never"     = "No arthritis",
         "prevalent" = "Arthritis at entry",
         "on_<60"    = "Onset <60",
         "on_60-74"  = "Onset 60\u201374",
         "on_>=75"   = "Onset \u226575")

SHAPE <- c("never" = 1, "prevalent" = 17, "on_<60" = 16, "on_60-74" = 16, "on_>=75" = 16)
LTY   <- c("never" = 2, "prevalent" = 1, "on_<60" = 1, "on_60-74" = 1, "on_>=75" = 1)
SIZE  <- c("never" = 0.55, "prevalent" = 0.8, "on_<60" = 0.75, "on_60-74" = 0.75, "on_>=75" = 0.75)

d$grp <- factor(d$grp, levels = c("never", "prevalent", "on_<60", "on_60-74", "on_>=75"))
d$band <- d$band + 2.5

p <- ggplot(d, aes(x = band, y = rate, colour = grp, shape = grp, linetype = grp, size = grp)) +
  geom_line(linewidth = 0.72) +
  geom_point(size = 2.1, stroke = 0.9) +
  scale_colour_manual(values = PAL, labels = LAB, name = NULL) +
  scale_shape_manual(values = SHAPE, labels = LAB, name = NULL) +
  scale_linetype_manual(values = LTY, labels = LAB, name = NULL) +
  scale_size_manual(values = SIZE, labels = LAB, name = NULL, guide = "none") +
  scale_y_log10(breaks = c(2, 5, 10, 20, 50, 100, 200, 300),
                labels = c(2, 5, 10, 20, 50, 100, 200, 300),
                limits = c(2.0, 380),
                expand = expansion(mult = c(0.01, 0.02))) +
  scale_x_continuous(breaks = seq(45, 95, 10), limits = c(43, 100),
                     expand = expansion(mult = c(0.01, 0.02))) +
  labs(x = "Attained age (years)",
       y = "Deaths per 1000 person-years",
       title = "Age-specific mortality by arthritis state, HRS 2000\u20132020",
       subtitle = "Lines span only the ages at which each group is observed; no curve is extrapolated",
       ) +
  theme_minimal(base_size = 9.2, base_family = "sans") +
  theme(
    plot.title = element_text(size = 10.4, face = "bold", colour = "#1A1D20",
                              margin = margin(b = 3)),
    plot.subtitle = element_text(size = 8.6, colour = "#5A6167", margin = margin(b = 10)),
    plot.title.position = "plot",
    axis.title.x = element_text(size = 9, margin = margin(t = 7)),
    axis.title.y = element_text(size = 9, margin = margin(r = 7)),
    axis.text = element_text(size = 8.4, colour = "#3A4045"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.28, colour = "#E3E6E8"),
    legend.position = c(0.025, 0.975),
    legend.justification = c(0, 1),
    legend.background = element_rect(fill = alpha("white", 0.88), colour = NA),
    legend.key.width = unit(1.35, "lines"),
    legend.key.height = unit(0.85, "lines"),
    legend.text = element_text(size = 8.2),
    legend.margin = margin(5, 8, 5, 9),
    plot.margin = margin(14, 16, 10, 12)
  )

ggsave("out/fig3_age_mortality_v18.png", p, width = 7.1, height = 4.55, dpi = 300, bg = "white")
ggsave("out/fig3_age_mortality_v18.pdf", p, width = 7.1, height = 4.55, device = cairo_pdf, bg = "white")
cat("OK fig3_v18\n")
