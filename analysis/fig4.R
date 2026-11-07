## Figure 4 (v2): functional limitations by attained age and age at onset
## 暴露改为 never + 三个 incident 起病层（基线已患病者不入图，见 Table 2）
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(patchwork); library(scales)
})
Sys.setlocale("LC_ALL", "en_US.UTF-8")
setwd(Sys.getenv("AGING_DB_ROOT", "."))

d <- read.csv("out/fig4_observed_states.csv", fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)
d$state <- trimws(d$state); d$band <- trimws(d$band)
d$band <- factor(d$band, levels = c("45-59","60-69","70-79","80-89","90+"))

map <- c("inc_<60" = "<60", "inc_60-74" = "60-74", "inc_>=75" = ">=75", "never" = "never")
d$og <- unname(map[d$state])
d <- d %>% filter(!is.na(og))

INK <- "#22262B"; GRID <- "#E9EDF0"
PAL <- c("<60" = "#1B3A5C", "60-74" = "#6E9BC0", ">=75" = "#B4CDDF", "never" = "#8A9096")
lab_of <- function(x) c("<60" = "<60", "60-74" = "60\u201374",
                        ">=75" = "\u226575", "never" = "No arthritis")[x]

mk <- function(df, ylab, ttl, sub, ymax) {
  df <- df %>% mutate(og = factor(og, levels = c("<60","60-74",">=75","never")))
  lv <- levels(droplevels(df$og))
  df$og <- factor(df$og, levels = lv,
                  labels = vapply(lv, lab_of, character(1), USE.NAMES = FALSE))
  cols <- setNames(PAL[lv], vapply(lv, lab_of, character(1), USE.NAMES = FALSE))
  ggplot(df, aes(x = band, y = mean, colour = og, group = og)) +
    geom_line(aes(linetype = og == "No arthritis"), linewidth = 0.55, show.legend = FALSE) +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.12, linewidth = 0.4) +
    geom_point(aes(shape = og == "No arthritis"), size = 2.1, stroke = 0.85, fill = "white") +
    scale_colour_manual(values = cols, name = "Age at onset") +
    scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 1), guide = "none") +
    scale_linetype_manual(values = c(`FALSE` = "solid", `TRUE` = "22"), guide = "none") +
    scale_y_continuous(limits = c(0, ymax), expand = expansion(mult = c(0.02, 0.02))) +
    labs(title = ttl, subtitle = sub, x = "Age band (years)", y = ylab) +
    theme_classic(base_size = 11, base_family = "sans") +
    theme(
      plot.title = element_text(face = "bold", size = 12, colour = INK, margin = margin(b = 6)),
      plot.subtitle = element_text(size = 9, colour = "#5A6066", margin = margin(b = 10)),
      axis.title = element_text(size = 9.8, colour = INK),
      axis.text = element_text(size = 9.6, colour = "#3A4046"),
      axis.line = element_line(colour = "#BCC2C7", linewidth = 0.4),
      axis.ticks = element_line(colour = "#BCC2C7", linewidth = 0.4),
      panel.grid.major.y = element_line(colour = GRID, linewidth = 0.4),
      legend.position = "top", legend.justification = "left",
      legend.title = element_text(size = 9.2, colour = "#3A4046"),
      legend.text = element_text(size = 9, colour = "#3A4046"),
      legend.key.width = unit(1.05, "lines"),
      legend.key.height = unit(0.9, "lines"),
      legend.margin = margin(0, 0, 0, 0),
      plot.margin = margin(8, 12, 8, 8)
    ) +
    guides(colour = guide_legend(nrow = 1, order = 1, title.position = "left"))
}

pA <- mk(d %>% filter(outcome == "ADL count"),
         "ADL limitations (0\u20136)", "A   HRS: ADL limitations",
         "Mean with 95% CI, by age at onset", 2.35)
pB <- mk(d %>% filter(outcome == "IADL count"),
         "IADL limitations (0\u20135)", "B   HRS: IADL limitations",
         "Mean with 95% CI, by age at onset", 2.20)
pC <- mk(d %>% filter(outcome == "Mobility limitation count"),
         "Limitations (0\u20132)", "C   SHARE: mobility limitations",
         "Mean with 95% CI, by age at onset", 1.65)

fig <- pA + pB + pC + plot_layout(widths = c(1, 1, 1.06), guides = "collect") &
  theme(legend.position = "bottom", legend.direction = "horizontal")

fig <- fig + plot_annotation(
  caption = paste0("Respondents who already had arthritis at their first interview are omitted here and reported in Table 2.\n",
                   "Strata are respondents whose arthritis was first reported during follow-up; age at onset is the midpoint of the\n",
                   "interval between the last negative and the first positive report. Panel C uses walking-100 m and climbing stairs."),
  theme = theme(plot.caption = element_text(size = 8.2, colour = "#6A7075",
                                            hjust = 0, margin = margin(t = 6, b = 2))))

ggsave("out/fig4_disability.png", fig, width = 12.8, height = 4.9, dpi = 320, bg = "white")
ggsave("out/fig4_disability.pdf", fig, width = 12.8, height = 4.9, device = cairo_pdf, bg = "white")
cat("OK fig4 v2\n")
