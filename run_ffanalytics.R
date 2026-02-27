#!/usr/bin/env Rscript

# Install and run ffanalytics package to get expert projections
# This script will scrape current fantasy football projections from multiple sources

# Set CRAN mirror
options(repos = c(CRAN = "https://cran.rstudio.com/"))

# Install required packages if not already installed
if (!require("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

if (!require("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}

if (!require("tidyr", quietly = TRUE)) {
  install.packages("tidyr")
}

# Install ffanalytics from the local directory
cat("Installing ffanalytics package...\n")
remotes::install_local("./ffanalytics-master", force = TRUE, quiet = FALSE)

# Load the package
library(ffanalytics)
library(dplyr)
library(tidyr)
library(lubridate)

# Set current season
current_season <- year(Sys.Date())

cat("Starting fantasy football projection scrape...\n")

# Define the sources to scrape (using reliable ones)
sources <- c("CBS", "ESPN", "FantasyPros", "NumberFire", "NFL")
positions <- c("QB", "RB", "WR", "TE")

# Scrape current season projections
cat("Scraping data from sources:", paste(sources, collapse = ", "), "\n")

tryCatch({
  # Scrape the data
  scraped_data <- scrape_data(
    src = sources,
    pos = positions,
    season = current_season  # Current season
    week = NULL     # Season-long projections
  )

  cat("Data scraping completed successfully!\n")

  # Calculate projections
  cat("Calculating projections...\n")
  projections <- projections_table(scraped_data)

  # Add additional information
  cat("Adding ECR, ADP, and player info...\n")
  final_projections <- projections %>%
    add_ecr() %>%
    add_adp() %>%
    add_player_info()

  # Save the results
  output_file <- sprintf("predictions/expert_projections_%d.csv", current_season)
  write.csv(final_projections, output_file, row.names = FALSE)
  cat("Expert projections saved to:", output_file, "\n")

  # Display summary
  cat("\n=== PROJECTION SUMMARY ===\n")
  cat("Total players projected:", nrow(final_projections), "\n")

  position_summary <- final_projections %>%
    group_by(pos) %>%
    summarise(
      count = n(),
      avg_points = round(mean(points, na.rm = TRUE), 1),
      .groups = 'drop'
    )

  print(position_summary)

}, error = function(e) {
  cat("Error occurred during scraping:\n")
  cat(as.character(e), "\n")
  cat("This might be due to website changes or network issues.\n")
  cat("Try running with fewer sources or check internet connection.\n")
})

cat("\nScript completed!\n")
