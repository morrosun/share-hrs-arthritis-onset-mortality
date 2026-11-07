## Figure 1 (v18): cohort flow + age at first diagnosis
## v18 改动：
##   (a) 流程改为最终队列 18,587（4,932 死亡），并把末段排除拆成 4 项
##       （无出生年 3 / 入组 <40 岁 9 / 2020 波 3,353 / 仅一次访谈且存活 5,312 = 8,677）；
##   (b) Panel B 的 SHARE 分布改用分析队列（18,587），题注同步。
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(patchwork); library(scales)
})
Sys.setlocale("LC_ALL", "en_US.UTF-8")
setwd(Sys.getenv("AGING_DB_ROOT", "."))

INK <- "#22262B"; GRID <- "#E9EDF0"

th <- theme_classic(base_size = 11, base_family = "sans") +
  theme(
    plot.title    = element_text(face = "bold", size = 12, colour = INK, margin = margin(b = 6)),
    plot.subtitle = element_text(size = 9, colour = "#5A6066", margin = margin(b = 10)),
    axis.title    = element_text(size = 9.8, colour = INK),
    axis.text     = element_text(size = 9.6, colour = "#3A4046"),
    axis.line     = element_line(colour = "#BCC2C7", linewidth = 0.4),
    axis.ticks    = element_line(colour = "#BCC2C7", linewidth = 0.4),
    panel.grid.major.y = element_line(colour = GRID, linewidth = 0.4),
    panel.grid.minor   = element_blank(),
    plot.margin   = margin(10, 12, 8, 8)
  )

## ---------------- Panel A: SHARE cohort construction ----------------
fb <- read.csv("out/fig1_flow_v18.csv", fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)
names(fb) <- c("lab", "n")
main <- fb
main$y <- c(9.2, 6.4, 3.6, 0.8)
main$txt1 <- c("Respondents in SHARE, waves 1\u20138",
               "Ever reported a physician diagnosis of arthritis",
               "Valid age at first diagnosis (5\u2013105 years)",
               "Analytic cohort")
main$txt2 <- c(format(main$n[1], big.mark = ","), format(main$n[2], big.mark = ","),
               format(main$n[3], big.mark = ","),
               paste0(format(main$n[4], big.mark = ","), "  (4,932 deaths)"))

exc <- data.frame(x = 6.4, y = c(7.8, 5.0, 2.95, 2.5, 2.05, 1.6, 1.15),
                  lab = c("\u2212 88,943 never reported arthritis",
                          "\u2212 20,636 no age at diagnosis",
                          "\u2212 8,677 in total:",
                          "3 no year of birth",
                          "9 entered before 40 years",
                          "3,353 entered in the 2020 wave",
                          "5,312 alive at their only interview"))

p1 <- ggplot() +
  geom_rect(data = main, aes(xmin = 0, xmax = 5.9, ymin = y - 0.95, ymax = y + 0.95),
            fill = "#F4F7F9", colour = "#9FB3C4", linewidth = 0.5) +
  geom_text(data = main, aes(x = 0.22, y = y + 0.38, label = txt1),
            hjust = 0, size = 3.15, colour = INK) +
  geom_text(data = main, aes(x = 0.22, y = y - 0.32, label = txt2),
            hjust = 0, size = 3.35, colour = "#2F5D8C", fontface = "bold") +
  geom_segment(data = main[-nrow(main), ], aes(x = 2.95, xend = 2.95, y = y - 1.0, yend = y - 1.75),
               arrow = arrow(length = unit(0.09, "inches"), type = "closed"),
               colour = "#8A9096", linewidth = 0.45) +
  geom_segment(data = main[1:3, ], aes(x = 5.9, xend = 6.3, y = y, yend = y),
               colour = "#C3CBD1", linewidth = 0.4) +
  geom_text(data = exc, aes(x = x, y = y, label = lab), hjust = 0, size = 2.9, colour = "#6A7075") +
  annotate("text", x = 2.95, y = 10.6, label = "A   Cohort construction",
           fontface = "bold", size = 4.2, colour = INK) +
  scale_x_continuous(limits = c(0, 12.6), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-0.4, 11.0), expand = c(0, 0)) +
  theme_void(base_size = 11) + theme(plot.margin = margin(10, 8, 8, 8))

## ---------------- Panel B: onset distribution ----------------
dd <- read.csv("out/fig1_onset_dist_v18.csv", fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)
dd <- dd %>% group_by(lib) %>% mutate(pct = 100 * n / sum(n)) %>% ungroup()

p2 <- ggplot(dd, aes(x = age, y = pct, colour = lib)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = 60, linetype = "22", colour = "#B0B7BD", linewidth = 0.45) +
  annotate("text", x = 61, y = max(dd$pct) * 0.96, label = "60 y", size = 2.9, colour = "#7A8085", hjust = 0) +
  scale_colour_manual(values = c(SHARE = "#3C6E9B", HRS = "#B0653A"), name = NULL) +
  scale_x_continuous(breaks = seq(20, 100, 10), limits = c(15, 105)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(title = "B   Distribution of age at first diagnosis",
       subtitle = "Percentage of each cohort per 5-year bin; HRS restricted to follow-up onsets",
       x = "Age at first diagnosis (years)", y = "Participants (%)") +
  th + theme(legend.position = c(0.16, 0.88), legend.background = element_blank(),
             legend.key.height = unit(0.85, "lines"))

f1 <- p1 + p2 + plot_layout(widths = c(1.30, 1)) +
  plot_annotation(caption = paste0("Panel A: exclusions are sequential and close exactly (3 + 9 + 3,353 + 5,312 = 8,677). ",
                                   "Panel B: SHARE uses the\nself-reported age at first diagnosis for every respondent in the analytic cohort (n = 18,587). ",
                                   "HRS shows all 7,252 respondents\nwith a first incident report, with onset taken as the midpoint of the last negative and first positive interviews; ",
                                   "respondents\naffected at entry have no recorded onset age and are not shown. HRS recruits at 50 years, so early onsets are rare."),
                  theme = theme(plot.caption = element_text(size = 8.0, colour = "#6A7075", hjust = 0,
                                                            margin = margin(t = 6, b = 2))))
ggsave("out/fig1_flow_onset_v18.png", f1, width = 12.8, height = 4.6, dpi = 320, bg = "white")
ggsave("out/fig1_flow_onset_v18.pdf", f1, width = 12.8, height = 4.6, device = cairo_pdf, bg = "white")
cat("OK fig1_v18\n")
