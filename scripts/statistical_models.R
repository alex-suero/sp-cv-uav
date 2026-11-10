# ==============================================================================
# Linear Mixed-Effects Models (LMMs), Repeated-Measures ANOVAs, and Broad-Sense
# Heritability (H^2) Analyses for Sweetpotato sUAS Phenotyping
# ==============================================================================
# Required packages: lme4, lmerTest, emmeans, multcomp
# Run from the repository root directory:
#   Rscript scripts/statistical_models.R
# ==============================================================================

library(lme4)
library(lmerTest)
library(emmeans)
library(multcomp)

# ------------------------------------------------------------------------------
# 1. Load 2026 Controlled Root Dataset (70 roots x 3 orientations = 210 views)
# ------------------------------------------------------------------------------
uav <- read.csv("data/2026_uav_test.csv")

# Build long dataframe for Two-Way Repeated-Measures ANOVA on Raw Volume APE
methods <- c("segmented_vol_cm3", "adj_sphere_vol_cm3", "sphere_vol_cm3", "ellip_vol_cm3", "adj_ellip_vol_cm3")
m_names <- c("Segmented", "Adj. Spheroid", "Spheroid", "Ellipsoid", "Adj. Ellipsoid")

recs <- list()
for (i in seq_along(methods)) {
    mcol <- methods[i]
    mname <- m_names[i]
    ape <- abs(uav[[mcol]] - uav$true_volume_cm3) / uav$true_volume_cm3 * 100
    recs[[i]] <- data.frame(
        Root_ID = as.factor(uav$root_uid),
        Test = as.factor(uav$test),
        Mask_Source = as.factor(ifelse(uav$dataset == "rcnn", "Mask R-CNN", "SAM 2")),
        Method = as.factor(mname),
        APE = ape,
        logAPE = log(ape + 0.1)
    )
}
vol_df <- do.call(rbind, recs)

cat("==============================================================================\n")
cat("1. TWO-WAY REPEATED-MEASURES ANOVA ON RAW VOLUMETRIC ERROR (APE %)\n")
cat("==============================================================================\n")
m_vol <- lmer(APE ~ Method * Mask_Source + (1|Root_ID), data=vol_df)
print(anova(m_vol))
print(cld(emmeans(m_vol, ~ Method), Letters=letters, adjust="tukey"))

# ------------------------------------------------------------------------------
# 2. Three-Level Density Parameterization ANOVA (Fixed, Pooled, Genotype-Specific)
# ------------------------------------------------------------------------------
rcnn <- subset(uav, dataset == "rcnn")

rcnn$loro_cult_dens <- 0
rcnn$loro_pool_dens <- 0
for (rid in unique(rcnn$root_uid)) {
    idx <- rcnn$root_uid == rid
    cln <- rcnn$clone_name[idx][1]
    rcnn$loro_cult_dens[idx] <- mean(rcnn$true_density_g_cm3[rcnn$clone_name == cln & !idx])
    rcnn$loro_pool_dens[idx] <- mean(rcnn$true_density_g_cm3[!idx])
}

top_methods <- c("segmented_vol_cm3", "adj_sphere_vol_cm3", "sphere_vol_cm3")
top_names <- c("Segmented", "Adj. Spheroid", "Spheroid")

recs_ind <- list()
recs_agg <- list()
k1 <- 1
k2 <- 1
for (i in seq_along(top_methods)) {
    mcol <- top_methods[i]
    mname <- top_names[i]
    pred_vol <- numeric(nrow(rcnn))
    for (rid in unique(rcnn$root_uid)) {
        idx <- rcnn$root_uid == rid
        fit <- lm(true_volume_cm3 ~ get(mcol), data=rcnn[!idx, ])
        pred_vol[idx] <- predict(fit, newdata=rcnn[idx, ])
    }
    rcnn$pred_vol <- pred_vol
    rcnn$w_fix <- rcnn$pred_vol * 1.05
    rcnn$w_pool <- rcnn$pred_vol * rcnn$loro_pool_dens
    rcnn$w_cult <- rcnn$pred_vol * rcnn$loro_cult_dens

    ape_fix <- abs(rcnn$w_fix - rcnn$true_weight_g) / rcnn$true_weight_g * 100
    ape_pool <- abs(rcnn$w_pool - rcnn$true_weight_g) / rcnn$true_weight_g * 100
    ape_cult <- abs(rcnn$w_cult - rcnn$true_weight_g) / rcnn$true_weight_g * 100

    recs_ind[[k1]] <- data.frame(Root_ID=rcnn$root_uid, Test=rcnn$test, Method=mname, Density="Fixed (1.05)", APE=ape_fix)
    recs_ind[[k1+1]] <- data.frame(Root_ID=rcnn$root_uid, Test=rcnn$test, Method=mname, Density="Pooled (0.98)", APE=ape_pool)
    recs_ind[[k1+2]] <- data.frame(Root_ID=rcnn$root_uid, Test=rcnn$test, Method=mname, Density="Genotype-Specific", APE=ape_cult)
    k1 <- k1 + 3

    agg <- aggregate(cbind(true_weight_g, w_fix, w_pool, w_cult) ~ clone_name + test, data=rcnn, FUN=sum)
    agg_ape_fix <- abs(agg$w_fix - agg$true_weight_g) / agg$true_weight_g * 100
    agg_ape_pool <- abs(agg$w_pool - agg$true_weight_g) / agg$true_weight_g * 100
    agg_ape_cult <- abs(agg$w_cult - agg$true_weight_g) / agg$true_weight_g * 100
    recs_agg[[k2]] <- data.frame(Plot_ID=agg$clone_name, Test=agg$test, Method=mname, Density="Fixed (1.05)", APE=agg_ape_fix)
    recs_agg[[k2+1]] <- data.frame(Plot_ID=agg$clone_name, Test=agg$test, Method=mname, Density="Pooled (0.98)", APE=agg_ape_pool)
    recs_agg[[k2+2]] <- data.frame(Plot_ID=agg$clone_name, Test=agg$test, Method=mname, Density="Genotype-Specific", APE=agg_ape_cult)
    k2 <- k2 + 3
}

df_ind <- do.call(rbind, recs_ind)
df_ind$Density <- factor(df_ind$Density, levels=c("Fixed (1.05)", "Pooled (0.98)", "Genotype-Specific"))
df_agg <- do.call(rbind, recs_agg)
df_agg$Density <- factor(df_agg$Density, levels=c("Fixed (1.05)", "Pooled (0.98)", "Genotype-Specific"))

cat("\n==============================================================================\n")
cat("2. THREE-LEVEL DENSITY ANOVA (LORO-CV): INDIVIDUAL ROOTS (n=210) & PLOTS (n=21)\n")
cat("==============================================================================\n")
m_ind <- lmer(APE ~ Method * Density + (1|Root_ID), data=df_ind)
print(anova(m_ind))
emm_ind <- emmeans(m_ind, ~ Density)
print(cld(emm_ind, Letters=letters, adjust="tukey"))

m_agg <- lmer(APE ~ Method * Density + (1|Plot_ID), data=df_agg)
print(anova(m_agg))
emm_agg <- emmeans(m_agg, ~ Density)
print(cld(emm_agg, Letters=letters, adjust="tukey"))

# ------------------------------------------------------------------------------
# 3. Orientation Repeatability (ICC) of Shape Traits on 2026 Controlled Dataset
# ------------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("3. ORIENTATION REPEATABILITY (ICC) ON 2026 CONTROLLED DATASET (70 roots x 3 views)\n")
cat("==============================================================================\n")
for (trait in c("std_diameter_cm", "straightness", "arc_length_cm", "max_diameter_cm")) {
    fr <- as.formula(paste(trait, "~ (1|root_uid)"))
    mr <- lmer(fr, data=rcnn)
    vcr <- as.data.frame(VarCorr(mr))
    v_id <- vcr$vcov[vcr$grp == "root_uid"]
    v_res <- vcr$vcov[vcr$grp == "Residual"]
    icc_1 <- v_id / (v_id + v_res)
    icc_3 <- v_id / (v_id + v_res / 3)
    cat(sprintf("Trait %-18s: Single-view ICC = %.3f | Three-view mean ICC = %.3f\n", trait, icc_1, icc_3))
}

# ------------------------------------------------------------------------------
# 4. 2024 Field Trial Broad-Sense Heritability (H^2) for Plot & Root Traits
# ------------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("4. 2024 FIELD TRIAL ENTRY-MEAN BROAD-SENSE HERITABILITY (H^2)\n")
cat("==============================================================================\n")
uav_plots <- read.csv("data/2024_uav_plots.csv")
s_cli <- read.csv("data/2024_sorter_CLI.csv")
s_cun <- read.csv("data/2024_sorter_CUN.csv")

cli_p <- subset(uav_plots, grepl("Clinton", location_name))
cun_p <- subset(uav_plots, grepl("Cunningham", location_name))
cli_m <- merge(cli_p, s_cli[, c("plot_number", "weight_all_kg", "mean_ld_all")], by="plot_number")
cun_m <- merge(cun_p, s_cun[, c("plot_number", "weight_all_kg", "mean_ld_all")], by="plot_number")
paired_plots <- rbind(cli_m, cun_m)

plots_per_g <- table(paired_plots$clone_name)
harm_n_L_plot <- length(plots_per_g) / sum(1.0 / plots_per_g)

for (info in list(
    c("weight_total_kg", "sUAS Total Yield (kg)"),
    c("weight_all_kg", "Optical Sorter Total Yield (kg)"),
    c("ld_ratio_avg", "sUAS Plot LD Ratio"),
    c("mean_ld_all", "Optical Sorter Plot LD Ratio")
)) {
    tcol <- info[1]
    tlabel <- info[2]
    mp <- lmer(as.formula(paste(tcol, "~ location_name + (1|clone_name)")), data=paired_plots)
    vc <- as.data.frame(VarCorr(mp))
    vg <- vc$vcov[vc$grp == "clone_name"]
    ve <- vc$vcov[vc$grp == "Residual"]
    h2_p <- vg / (vg + ve / harm_n_L_plot)
    cat(sprintf("%-34s: Entry-mean H^2 = %.3f\n", tlabel, h2_p))
}

df_roots <- read.csv("data/2024_uav_roots.csv")
df_roots <- df_roots[!is.na(df_roots$straightness) & !is.na(df_roots$std_diameter_cm), ]
df_roots$loc_plot <- paste0(df_roots$location_name, "_", df_roots$plot_number)

roots_per_g <- table(df_roots$clone_name)
harm_n_R <- length(roots_per_g) / sum(1.0 / roots_per_g)
plots_per_g_df <- aggregate(loc_plot ~ clone_name, data=df_roots, FUN=function(x) length(unique(x)))
harm_n_L <- nrow(plots_per_g_df) / sum(1.0 / plots_per_g_df$loc_plot)

for (trait in c("std_diameter_cm", "ld_ratio", "straightness")) {
    f1 <- as.formula(paste(trait, "~ location_name + (1|clone_name) + (1|loc_plot)"))
    m1 <- lmer(f1, data=df_roots)
    vc <- as.data.frame(VarCorr(m1))
    vg <- vc$vcov[vc$grp == "clone_name"]
    vp <- vc$vcov[vc$grp == "loc_plot"]
    ve <- vc$vcov[vc$grp == "Residual"]
    H2_harm <- vg / (vg + vp / harm_n_L + ve / harm_n_R)
    cat(sprintf("Root-level %-23s: Entry-mean H^2 = %.3f (Vg=%.5f, Vp=%.5f, Ve=%.5f)\n",
                trait, H2_harm, vg, vp, ve))
}
