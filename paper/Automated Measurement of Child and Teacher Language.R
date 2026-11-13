# =============================================================================
# Project: Automated Measurement of Child and Teacher Language in Preschool Classrooms
# Authors: Elbaum et al
# Date: November 2026
# =============================================================================
# --------------------------------------------------------------------
# Table of Contents
# --------------------------------------------------------------------
# 1. Set-up
# 2. Load in & prepare data
# 3. Aggregated data
#    3.1 Child-level aggregated dataset
#    3.2 Semester MLUs (Fall & Spring)
# 4. Analyses
#    4.1 MLU Over Time
#        4.1.1 Outcome: Child MLU
#        4.1.2 Outcome: Teacher MLU
#    4.2 Language Outcomes (PLS-5 Models)
#        4.2.1 Child MLU predicting PLS-5
#        4.2.2 Teacher MLU predicting PLS-5
#    4.3 Mediation Analyses
# --------------------------------------------------------------------
# =============================================================================
# 1. Set-up
# =============================================================================

# Reproducibility
options(scipen = 999)
set.seed(20261103)

# Load packages
suppressPackageStartupMessages({
  library(tidyverse)   # dplyr, tidyr, ggplot2, purrr
  library(lme4)        # Mixed-effects modeling
  library(lmerTest)    # Satterthwaite approximations for LMMs
  library(performance) # ICC and model diagnostics
  library(psych)       # Descriptive statistics & psych::mediate
  library(mediation)   # Causal mediation & moderated mediation
  library(effectsize)  # Effect size calculations
  library(EMAtools)    # Another effect size package
})

# Function to convert lm t-values to Cohen's d effect sizes
calc_lm_d <- function(model) {
  t_vals <- coef(summary(model))[, "t value"]
  df_res <- df.residual(model)
  return((2 * t_vals) / sqrt(df_res))
}
# =============================================================================
# 2. Load in & prepare data
# =============================================================================

  # Load in data
  df <- read.csv((file.path("auto.meas.child.teach.csv")), header = TRUE, stringsAsFactors = FALSE)
  
  # Hearing status coding / Reference = Hearing Loss (HL)
  df <- df %>%
    mutate(
      hearingStatus = factor(hearingStatus),
      hearingStatus = relevel(hearingStatus, ref = "HL")
    )
  
  # Contrast/effect coding for 3-level cohort variable (Cohort 1 reference: -1, -1)
  df <- df %>%
    mutate(
      cohortCode1 = case_when(cohort == "1" ~ -1, cohort == "2" ~  1, cohort == "3" ~  0),
      cohortCode2 = case_when(cohort == "1" ~ -1, cohort == "2" ~  0, cohort == "3" ~  1)
    )

# =============================================================================
# 3. Aggregated data
# =============================================================================
  # -----------------------------------------------------------------------------
  # 3.1 Child-level aggregated dataset
  # -----------------------------------------------------------------------------
    child <- df %>%
      group_by(Subject_ID) %>%
      summarise(
        hearingStatus        = first(hearingStatus),
        cohort               = first(cohort),
        cohortCode1          = first(cohortCode1),
        cohortCode2          = first(cohortCode2),
        timeBetweenPLSMonths = first(timeBetweenPLSMonths),
        across(
          all_of(c(
            "mluKCHI", "mluFEM",
            "AC_Raw_Pre", "AC_Standard_Pre", "EC_Raw_Pre", "EC_Standard_Pre",
            "AC_Raw_Post", "AC_Standard_Post", "EC_Raw_Post", "EC_Standard_Post"
          )),
          ~ mean(.x, na.rm = TRUE)
        ),
        .groups = "drop"
      )
    
    # Effect coding for hearing status
    child <- child %>%
      mutate(hearingStatus_effect = factor(hearingStatus, levels = c("TH", "HL")))
    contrasts(child$hearingStatus_effect) <- contr.sum(2)
  
  # -----------------------------------------------------------------------------
  # 3.2 Semester MLUs (Fall vs. Spring - for mediation)
  # -----------------------------------------------------------------------------
    semester <- df %>%
      filter(semester %in% c("fall", "spring")) %>%
      group_by(Subject_ID, semester) %>%
      summarise(
        teacher = mean(mluFEM, na.rm = TRUE),
        child   = mean(mluKCHI, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      pivot_wider(
        names_from  = semester,
        values_from = c(teacher, child),
        names_glue  = "{.value}{str_to_title(semester)}"
      )
    
    # Merge semester into child dataset
    child <- left_join(child, semester, by = "Subject_ID")
    

# =============================================================================
# 4. Analyses
# =============================================================================
  # -----------------------------------------------------------------------------
  # 4.1 Child MLU Over Time
  # -----------------------------------------------------------------------------

    # --- 4.1.1 Outcome: Child MLU (mluKCHI) ---
      # Descriptives
      describe(child$mluKCHI)
      describeBy(child$mluKCHI, group = child$hearingStatus)
      
      # Null model (ICC)
      ml_null_child <- lmer(mluKCHI ~ (1 | Subject_ID), data = df)
      summary(ml_null_child)
      icc(ml_null_child)
      
      # Conditional growth model
      ml_final_child <- lmer(
        mluKCHI ~ timeInStudyMonths + hearingStatus + cohortCode1 + cohortCode2 + mluFEM + (1 | Subject_ID),
        data = df
      )
      summary(ml_final_child)
      lme.dscore(ml_final_child,data=df, type="lme4")


    # --- 4.1.2 Outcome: Teacher MLU (mluFEM) ---
      # Descriptives
      describe(child$mluFEM)
      describeBy(child$mluFEM, group = child$hearingStatus)
      corr.test(df$mluKCHI, df$mluFEM)
      
      # Null model (Unconditional ICC)
      ml_null_teach <- lmer(mluFEM ~ (1 | Subject_ID), data = df)
      summary(ml_null_teach)
      icc(ml_null_teach)
      
      # Random intercept model (singularity error; documented in Supplemental Mats)
      teach_growth_hlm <- lmer(
        mluFEM ~ mluKCHI + hearingStatus + timeInStudyMonths + cohortCode1 + cohortCode2 + (1 | Subject_ID),
        data = df
      )
      summary(teach_growth_hlm)
      
      # More parsimonious fixed-effects OLS model (no child-level nesting)
      teach_growth_lm <- lm(
        mluFEM ~ mluKCHI + hearingStatus + timeInStudyMonths + cohortCode1 + cohortCode2,
        data = df
      )
      summary(teach_growth_lm)
      calc_lm_d(teach_growth_lm)

  # -----------------------------------------------------------------------------
  # 4.2 Language Outcomes (PLS-5 Models)
  # -----------------------------------------------------------------------------
    # --- 4.2.1 Child MLU predicting PLS-5 ---

      # Baseline hearing status models (Standard Scores)
      pls_base_ec <- lm(EC_Standard_Post ~ hearingStatus + cohortCode1 + cohortCode2, data = child)
      summary(pls_base_ec)
      calc_lm_d(pls_base_ec)
      
      pls_base_ac <- lm(AC_Standard_Post ~ hearingStatus + cohortCode1 + cohortCode2, data = child)
      summary(pls_base_ac)
      calc_lm_d(pls_base_ac)
      
      # Standard score models
      pls_ss_ec_child <- lm(EC_Standard_Post ~ mluKCHI + hearingStatus + cohortCode1 + cohortCode2, data = child)
      summary(pls_ss_ec_child)
      calc_lm_d(pls_ss_ec_child)
      
      pls_ss_ac_child <- lm(AC_Standard_Post ~ mluKCHI + hearingStatus + cohortCode1 + cohortCode2, data = child)
      summary(pls_ss_ac_child)
      calc_lm_d(pls_ss_ac_child)
      
      # Raw score models (controlling for fall raw score and testing interval)
      pls_raw_ec_child <- lm(
        EC_Raw_Post ~ mluKCHI + hearingStatus + EC_Raw_Pre + timeBetweenPLSMonths + cohortCode1 + cohortCode2,
        data = child)
      summary(pls_raw_ec_child)
      calc_lm_d(pls_raw_ec_child)
      
      pls_raw_ac_child <- lm(
        AC_Raw_Post ~ mluKCHI * hearingStatus + AC_Raw_Pre + timeBetweenPLSMonths + cohortCode1 + cohortCode2,
        data = child)
      summary(pls_raw_ac_child)
      calc_lm_d(pls_raw_ac_child)
      
        # Effect-coded interaction decomposition for Auditory Comprehension
        pls_raw_ac_effect <- lm(
          AC_Raw_Post ~ mluKCHI * hearingStatus_effect + AC_Raw_Pre + timeBetweenPLSMonths + cohortCode1 + cohortCode2,
          data = child)
        summary(pls_raw_ac_effect)
        calc_lm_d(pls_raw_ac_effect)

    # --- 4.2.2 Teacher MLU predicting PLS-5 ---

      # Standard score models
      pls_ss_ec_teach <- lm(EC_Standard_Post ~ mluFEM + hearingStatus + cohortCode1 + cohortCode2, data = child)
      summary(pls_ss_ec_teach)
      calc_lm_d(pls_ss_ec_teach)
      
      pls_ss_ac_teach <- lm(AC_Standard_Post ~ mluFEM * hearingStatus + cohortCode1 + cohortCode2, data = child)
      summary(pls_ss_ac_teach)
      calc_lm_d(pls_ss_ac_teach)
      
        # Effect-coded version for Auditory Comprehension
        pls_ss_ac_teach_effect <- lm(AC_Standard_Post ~ mluFEM * hearingStatus_effect + cohortCode1 + cohortCode2, data = child)
        summary(pls_ss_ac_teach_effect)
        calc_lm_d(pls_ss_ac_teach_effect)
      
      # Raw score models
      pls_raw_ec_teach <- lm(
        EC_Raw_Post ~ mluFEM + hearingStatus + EC_Raw_Pre + timeBetweenPLSMonths + cohortCode1 + cohortCode2,
        data = child)
      summary(pls_raw_ec_teach)
      calc_lm_d(pls_raw_ec_teach)
      
      pls_raw_ac_teach <- lm(
        AC_Raw_Post ~ mluFEM + hearingStatus + AC_Raw_Pre + timeBetweenPLSMonths + cohortCode1 + cohortCode2,
        data = child)
      summary(pls_raw_ac_teach)
      calc_lm_d(pls_raw_ac_teach)

  # -----------------------------------------------------------------------------
  # 4.3 Mediation Analyses
  # -----------------------------------------------------------------------------
    # Teacher mean MLU (fall) -> child mean MLU spring -> Spring PLS

    # Simple mediation models via psych package (Teacher Fall -> Child Spring -> Spring PLS)
    med_psych_ac <- psych::mediate(
      AC_Standard_Post ~ teacherFall + cohortCode1 + cohortCode2 + (childSpring),
      data = child, n.iter = 5000, zero = FALSE)
    summary(med_psych_ac)
    
    med_psych_ec <- psych::mediate(
      EC_Standard_Post ~ teacherFall + cohortCode1 + cohortCode2 + (childSpring),
      data = child, n.iter = 5000, zero = FALSE)
    summary(med_psych_ec)


    # Moderated mediation check: Auditory Comprehension (AC)
    complete_cases_ac <- child %>%
      drop_na(teacherFall, childSpring, AC_Standard_Post, hearingStatus, cohortCode1, cohortCode2)
    
    med_m_ac <- lm(childSpring ~ teacherFall * hearingStatus + cohortCode1 + cohortCode2, data = complete_cases_ac)
    med_y_ac <- lm(
      AC_Standard_Post ~ teacherFall * hearingStatus + childSpring * hearingStatus + cohortCode1 + cohortCode2,
      data = complete_cases_ac)
    
    fit_med_ac <- mediation::mediate(med_m_ac, med_y_ac, treat = "teacherFall", mediator = "childSpring", sims = 5000)
    summary(fit_med_ac)
    
    mediation::test.modmed(
      fit_med_ac,
      covariates.1 = list(hearingStatus = "TH"),
      covariates.2 = list(hearingStatus = "HL"),
      sims = 5000)
    
    # Moderated mediation check: Expressive Communication (EC)
    complete_cases_ec <- child %>%
      drop_na(teacherFall, childSpring, EC_Standard_Post, hearingStatus, cohortCode1, cohortCode2)
    
    med_m_ec <- lm(childSpring ~ teacherFall * hearingStatus + cohortCode1 + cohortCode2, data = complete_cases_ec)
    med_y_ec <- lm(
      EC_Standard_Post ~ teacherFall * hearingStatus + childSpring * hearingStatus + cohortCode1 + cohortCode2,
      data = complete_cases_ec)
    
    fit_med_ec <- mediation::mediate(med_m_ec, med_y_ec, treat = "teacherFall", mediator = "childSpring", sims = 5000)
    summary(fit_med_ec)
    
    mediation::test.modmed(
      fit_med_ec,
      covariates.1 = list(hearingStatus = "TH"),
      covariates.2 = list(hearingStatus = "HL"),
      sims = 5000)

