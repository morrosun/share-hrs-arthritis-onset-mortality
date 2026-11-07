## Figure 1: crude mortality vs age-adjusted hazard (SHARE)
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(patchwork); library(scales)
})
setwd(Sys.getenv("AGING_DB_ROOT", "."))

g <- read.csv("out/fig1_groups.csv", fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)
h <- read.csv("out/fig1_hr.csv",     fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)

g$grp <- trimws(g$grp); h$grp <- trimws(h$grp)
lv   <- c("<60", "60-74", ">=75")
labs <- c("<60", "60\u201374", "\u226575")

g$xl <- factor(labs[match(g$grp, lv)], levels = labs)
h$xl <- factor(labs[match(h$grp, lv)], levels = rev(labs))
h$isref <- h$grp == "<60"

BLUE <- "#3C6E9B"; GREY <- "#8A9096"; INK <- "#22262B"; GRID <- "#E9EDF0"

th <- theme_classic(base_size = 11, base_family = "sans") +
  theme(
    plot.title    = element_text(face = "bold", size = 12, colour = INK, margin = margin(b = 6)),
    plot.subtitle = element_text(size = 9, colour = "#5A6066", margin = margin(b = 12)),
    axis.title    = element_text(size = 9.8, colour = INK),
    axis.text     = element_text(size = 10, colour = "#3A4046"),
    axis.line     = element_line(colour = "#BCC2C7", linewidth = 0.4),
    axis.ticks    = element_line(colour = "#BCC2C7", linewidth = 0.4),
    panel.grid.major.y = element_line(colour = GRID, linewidth = 0.4),
    panel.grid.minor   = element_blank(),
    plot.margin   = margin(10, 10, 6, 6)
  )

## ---- A: crude mortality rate ----
pA <- ggplot(g, aes(x = xl, y = rate)) +
  geom_col(width = 0.56, fill = BLUE, alpha = 0.92) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.13,
                colour = INK, linewidth = 0.5) +
  geom_text(aes(y = hi + 3.2, label = sprintf("%.1f", rate)),
            size = 3.25, colour = INK) +
  scale_y_continuous(limits = c(0, 134), breaks = seq(0, 120, 30),
                     expand = expansion(mult = c(0, 0))) +
  labs(title = "A   Crude mortality",
       subtitle = "Bars: rate; whiskers: 95% CI",
       x = NULL, y = "Deaths / 1000 person-years") +
  th

## ---- B: age at cohort entry ----
pB <- ggplot(g, aes(x = xl, y = entry_mean)) +
  geom_errorbar(aes(ymin = entry_mean - entry_sd, ymax = entry_mean + entry_sd),
                width = 0.13, colour = "#6E747A", linewidth = 0.6) +
  geom_point(size = 3.3, shape = 21, fill = "white", colour = "#6E747A", stroke = 1.1) +
  geom_text(aes(y = entry_mean + entry_sd + 2.4, label = sprintf("%.1f", entry_mean)),
            size = 3.25, colour = INK) +
  scale_y_continuous(limits = c(44, 100), breaks = seq(50, 90, 10),
                     expand = expansion(mult = c(0, 0))) +
  labs(title = "B   Age at cohort entry",
       subtitle = "Mean and SD; 20-year span across groups",
       x = NULL, y = "Age at entry (years)") +
  th

## ---- C: adjusted HR forest ----
pC <- ggplot(h, aes(x = hr, y = xl)) +
  geom_vline(xintercept = 1, linetype = "22", colour = GREY, linewidth = 0.45) +
  geom_errorbar(aes(xmin = lo, xmax = hi), orientation = "y", width = 0.14,
                colour = ifelse(h$isref, GREY, BLUE), linewidth = 0.7) +
  geom_point(aes(fill = isref), shape = 22, size = 3.0,
             colour = ifelse(h$isref, GREY, BLUE), stroke = 0.9) +
  geom_text(aes(x = 1.21,
                label = ifelse(isref, "1.00 (ref)",
                               sprintf("%.2f (%.2f\u2013%.2f)", hr, lo, hi))),
            hjust = 0, size = 3.1, colour = INK) +
  scale_fill_manual(values = c(`FALSE` = BLUE, `TRUE` = "white"), guide = "none") +
  scale_x_log10(limits = c(0.60, 1.50), breaks = c(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
                labels = function(x) sprintf("%.1f", x)) +
  labs(title = "C   Adjusted hazard ratio",
       subtitle = "Age time scale, left-truncated; ref = onset <60",
       x = "Hazard ratio for all-cause death (log scale)", y = NULL) +
  th +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(colour = GRID, linewidth = 0.4),
        axis.text.y = element_text(size = 10.3, colour = INK),
        plot.margin = margin(10, 4, 6, 10))

fig <- pA + pB + pC + plot_layout(widths = c(1, 1, 1.30)) +
  plot_annotation(
    caption = paste0("SHARE cohort, n = 18,530; 4,875 deaths. Panel C: Cox model with age as the time scale and ",
                     "delayed entry at cohort entry, adjusted for sex, education and smoking."),
    theme = theme(plot.caption = element_text(size = 8.2, colour = "#6A7075",
                                              hjust = 0, margin = margin(t = 4, b = 2))))

ggsave("out/fig2_crude_vs_adjusted.png", fig, width = 12.6, height = 4.5,
       dpi = 320, bg = "white")
ggsave("out/fig2_crude_vs_adjusted.pdf", fig, width = 12.6, height = 4.5,
       device = cairo_pdf, bg = "white")
cat("OK fig1\n")
