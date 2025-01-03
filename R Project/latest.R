library(shiny)
library(shinydashboard)
library(readr)
library(dplyr)
library(caret)
library(randomForest)
library(ggplot2)
library(plotly)
library(pROC)

setwd("C:\Users\dell\Desktop\r-project")

# Load and preprocess the data
telco_data <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

telco_data <- telco_data %>%
  select(tenure, MonthlyCharges, Contract, PaymentMethod, InternetService, Churn) %>%
  mutate(
    Churn = as.factor(Churn),
    Contract = as.factor(Contract),
    PaymentMethod = as.factor(PaymentMethod),
    InternetService = as.factor(InternetService)
  ) %>%
  mutate(
    tenure = ifelse(tenure < 0, NA, tenure), 
    MonthlyCharges = ifelse(MonthlyCharges < 0, NA, MonthlyCharges)
  ) %>%
  mutate_if(is.numeric, ~ifelse(is.na(.), mean(., na.rm = TRUE), .))

# Split data into training and test sets
set.seed(123)
trainIndex <- createDataPartition(telco_data$Churn, p = 0.8, list = FALSE, times = 1)
train_data <- telco_data[trainIndex, ]
test_data <- telco_data[-trainIndex, ]

# Define UI
ui <- dashboardPage(
  dashboardHeader(title = "Telco Customer Churn Prediction"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Intro", tabName = "intro", icon = icon("info-circle")),
      menuItem("Data Summary", tabName = "data_summary", icon = icon("table")),
      menuItem("Model Summary", tabName = "model_summary", icon = icon("chart-bar")),
      menuItem("Prediction", tabName = "prediction", icon = icon("calculator")),
      menuItem("Visualizations", tabName = "visualizations", icon = icon("chart-line")),
      menuItem("Feature Importance", tabName = "feature_importance", icon = icon("sliders-h")),
      menuItem("ROC Curve", tabName = "roc_curve", icon = icon("signal"))
    )
  ),
  dashboardBody(
    tags$head(
      tags$link(rel = "stylesheet", type = "text/css", href = "styles.css"),
      tags$script(src = "script.js")
    ),
    tabItems(
      tabItem(
        tabName = "intro",
        fluidRow(
          column(12, align = "left",
                 h1("Welcome to Telco Customer Churn Prediction"),
                 p("This dashboard predicts customer churn for a telecommunications company using advanced machine learning models."),
                 h3("Key Objectives:"),
                 tags$ul(
                   tags$li("Analyze customer behavior to predict churn."),
                   tags$li("Understand the factors influencing customer churn."),
                   tags$li("Provide actionable insights for business decision-making.")
                 )
          )
        )
      ),
      tabItem(tabName = "data_summary", verbatimTextOutput("data_summary")),
      tabItem(
        tabName = "model_summary",
        actionButton("train_models", "Train Models"),
        verbatimTextOutput("model_summary")
      ),
      tabItem(
        tabName = "prediction",
        sidebarLayout(
          sidebarPanel(
            numericInput("tenure", "Tenure (months)", value = 12, min = 0),
            numericInput("MonthlyCharges", "Monthly Charges", value = 50, min = 0),
            selectInput("Contract", "Contract Type", choices = levels(telco_data$Contract)),
            selectInput("PaymentMethod", "Payment Method", choices = levels(telco_data$PaymentMethod)),
            selectInput("InternetService", "Internet Service", choices = levels(telco_data$InternetService)),
            actionButton("predict", "Predict Churn")
          ),
          mainPanel(verbatimTextOutput("predictions"))
        )
      ),
      tabItem(
        tabName = "visualizations",
        tabsetPanel(
          tabPanel("2D Visualizations", plotlyOutput("tenure_dist"), plotlyOutput("monthly_charges_dist"), plotlyOutput("churn_bar")),
          tabPanel("3D Visualizations", plotlyOutput("scatter_3d"), plotlyOutput("surface_plot"), plotlyOutput("boxplot_3d"))
        )
      ),
      tabItem(tabName = "feature_importance", plotlyOutput("feature_importance")),
      tabItem(tabName = "roc_curve", plotlyOutput("roc_curve"))
    )
  )
)

# Define Server
server <- function(input, output, session) {
  models <- reactiveValues(logistic = NULL, rf = NULL, logistic_pred_prob = NULL, rf_pred_prob = NULL)
  
  # Data Summary
  output$data_summary <- renderPrint({
    list(
      Structure = capture.output(str(telco_data)),
      Summary = summary(telco_data),
      Missing_Values = colSums(is.na(telco_data))
    )
  })
  
  # Train Models
  observeEvent(input$train_models, {
    models$logistic <- glm(Churn ~ ., data = train_data, family = binomial)
    models$rf <- randomForest(Churn ~ ., data = train_data, ntree = 100, mtry = 2)
    models$logistic_pred_prob <- predict(models$logistic, test_data, type = "response")
    models$rf_pred_prob <- predict(models$rf, test_data, type = "prob")[, 2]
    
    output$model_summary <- renderPrint({
      list(
        `Logistic Regression` = summary(models$logistic),
        `Random Forest` = models$rf
      )
    })
  })
  
  # Predictions
  observeEvent(input$predict, {
    if (is.null(models$logistic) || is.null(models$rf)) {
      output$predictions <- renderPrint({ "Please train the models before making predictions." })
      return()
    }
    prediction_data <- data.frame(
      tenure = input$tenure,
      MonthlyCharges = input$MonthlyCharges,
      Contract = factor(input$Contract, levels = levels(train_data$Contract)),
      PaymentMethod = factor(input$PaymentMethod, levels = levels(train_data$PaymentMethod)),
      InternetService = factor(input$InternetService, levels = levels(train_data$InternetService))
    )
    logistic_pred <- predict(models$logistic, prediction_data, type = "response")
    rf_pred <- predict(models$rf, prediction_data)
    output$predictions <- renderPrint({
      list(
        `Logistic Prediction` = ifelse(logistic_pred > 0.5, "Yes", "No"),
        `Logistic Probability` = round(logistic_pred, 4),
        `Random Forest Prediction` = rf_pred
      )
    })
  })
  
  # Feature Importance
  output$feature_importance <- renderPlotly({
    if (inherits(models$rf, "randomForest")) {
      imp <- importance(models$rf)
      feature_importance_df <- data.frame(Feature = rownames(imp), Importance = imp[, 1])
      plot_ly(feature_importance_df, x = ~Feature, y = ~Importance, type = "bar")
    }
  })
  
  # ROC Curve
  output$roc_curve <- renderPlotly({
    if (!is.null(models$logistic_pred_prob) && !is.null(models$rf_pred_prob)) {
      logistic_roc <- roc(test_data$Churn, models$logistic_pred_prob)
      rf_roc <- roc(test_data$Churn, models$rf_pred_prob)
      logistic_roc_df <- data.frame(FPR = logistic_roc$specificities, TPR = logistic_roc$sensitivities, Model = "Logistic Regression")
      rf_roc_df <- data.frame(FPR = rf_roc$specificities, TPR = rf_roc$sensitivities, Model = "Random Forest")
      roc_data <- rbind(logistic_roc_df, rf_roc_df)
      ggplotly(ggplot(roc_data, aes(x = 1 - FPR, y = TPR, color = Model)) +
                 geom_line(size = 1) +
                 theme_minimal() +
                 labs(title = "ROC Curve", x = "False Positive Rate", y = "True Positive Rate"))
    }
  })
  
  # 2D Visualizations
  output$tenure_dist <- renderPlotly({
    p <- ggplot(telco_data, aes(x = tenure, fill = Churn)) +
      geom_histogram(bins = 30, position = "dodge") +
      labs(title = "Distribution of Tenure", x = "Tenure (months)", y = "Count") +
      theme_minimal()
    ggplotly(p)
  })
  output$monthly_charges_dist <- renderPlotly({
    p <- ggplot(telco_data, aes(x = MonthlyCharges, fill = Churn)) +
      geom_histogram(bins = 30, position = "dodge") +
      labs(title = "Distribution of Monthly Charges", x = "Monthly Charges", y = "Count") +
      theme_minimal()
    ggplotly(p)
  })
  output$churn_bar <- renderPlotly({
    p <- ggplot(telco_data, aes(x = Churn)) +
      geom_bar(fill = c("salmon", "skyblue")) +
      labs(title = "Churn Count", x = "Churn", y = "Count") +
      theme_minimal()
    ggplotly(p)
  })
  
  # 3D Visualizations
  output$scatter_3d <- renderPlotly({
    plot_ly(telco_data, x = ~tenure, y = ~MonthlyCharges, z = ~as.numeric(Churn), 
            color = ~Churn, type = "scatter3d", mode = "markers")
  })
  output$surface_plot <- renderPlotly({
    plot_ly(z = ~volcano, type = "surface")
  })
  output$boxplot_3d <- renderPlotly({
    plot_ly(telco_data, x = ~tenure, y = ~MonthlyCharges, z = ~as.numeric(Churn), 
            type = "box", color = ~Churn)
  })
}

# Run App
shinyApp(ui, server)
