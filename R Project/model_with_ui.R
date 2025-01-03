library(shiny)
library(shinyjs)
library(readr)
library(dplyr)
library(caret)
library(randomForest)
library(ggplot2)
library(plotly)
library(pROC)
setwd("C:/Users/harsh/OneDrive/Desktop/r-project")

# Load and preprocess data
telco_data <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
telco_data <- telco_data %>%
  select(tenure, MonthlyCharges, Contract, PaymentMethod, InternetService, Churn) %>%
  mutate(
    Churn = as.factor(Churn),
    Contract = as.factor(Contract),
    PaymentMethod = as.factor(PaymentMethod),
    InternetService = as.factor(InternetService),
    tenure = ifelse(tenure < 0, NA, tenure),
    MonthlyCharges = ifelse(MonthlyCharges < 0, NA, MonthlyCharges)
  ) %>%
  mutate_if(is.numeric, ~ifelse(is.na(.), mean(., na.rm = TRUE), .))

set.seed(123)
trainIndex <- createDataPartition(telco_data$Churn, p = 0.8, list = FALSE, times = 1)
train_data <- telco_data[trainIndex, ]
test_data <- telco_data[-trainIndex, ]

# Define UI
ui <- fluidPage(
  useShinyjs(), # For smooth scrolling
  tags$head(
    tags$style(HTML("
      body { 
        font-family: 'Arial', sans-serif; 
        background-color: #121212; 
        color: #ffffff;
        margin: 0;
        overflow-x: hidden;
      }
      .section {
        min-height: 100vh;
        padding: 50px;
        opacity: 0;
        transform: scale(0.9);
        transition: opacity 1s, transform 1s;
      }
      .section.visible {
        opacity: 1;
        transform: scale(1);
      }
    .landing-page {
    background: url('www/wp8574115-4k-black-oled-wallpapers.jpg') no-repeat center center fixed;
    background-size: cover;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
      .landing-page h1 {
        font-size: 3rem;
        color: #f83949;
      }
      .landing-page p {
        font-size: 1.2rem;
        margin: 20px 0;
      }
      .scroll-indicator {
        font-size: 1rem;
        color: #086bfd;
        margin-top: 20px;
        cursor: pointer;
      }
      .section h2 {
        font-size: 2rem;
        margin-bottom: 20px;
        color: #f83949;
      }
      .btn-primary {
        background-color: #086bfd;
        border: none;
        color: white;
        padding: 10px 20px;
        cursor: pointer;
        border-radius: 5px;
        margin: 20px 0;
      }
    "))
  ),
  
  # JavaScript for triggering animations on scroll
  tags$script(HTML("
    document.addEventListener('DOMContentLoaded', function() {
      const sections = document.querySelectorAll('.section');
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          }
        });
      }, { threshold: 0.2 });
      sections.forEach(section => observer.observe(section));
    });
  ")),
  
  # Landing Page Section
  div(
    class = "section landing-page",
    h1("Customer Churn Prediction"),
    p("Welcome to the Customer Churn Prediction Dashboard! Scroll down to explore data insights, model summaries, and predictions."),
    div(
      class = "scroll-indicator",
      onclick = "document.getElementById('data-summary').scrollIntoView({behavior: 'smooth'});",
      "↓ Scroll to Explore"
    )
  ),
  
  # Data Summary Section
  div(
    id = "data-summary",
    class = "section",
    h2("Data Summary"),
    verbatimTextOutput("summary")
  ),
  
  # Model Summary Section
  div(
    id = "model-summary",
    class = "section",
    h2("Model Summary"),
    verbatimTextOutput("model_summary"),
    actionButton("train_models", "Train Models", class = "btn-primary")
  ),
  
  # Predictions Section
  div(
    id = "predictions",
    class = "section",
    h2("Predictions"),
    numericInput("tenure", "Tenure (months)", value = 12, min = 0),
    numericInput("MonthlyCharges", "Monthly Charges", value = 50, min = 0),
    selectInput("Contract", "Contract Type", choices = unique(telco_data$Contract)),
    selectInput("PaymentMethod", "Payment Method", choices = unique(telco_data$PaymentMethod)),
    selectInput("InternetService", "Internet Service", choices = unique(telco_data$InternetService)),
    actionButton("predict", "Predict Churn", class = "btn-primary"),
    verbatimTextOutput("predictions")
  ),
  
  # Visualizations Section
  div(
    id = "visualizations",
    class = "section",
    h2("Visualizations"),
    plotlyOutput("tenure_dist"),
    plotlyOutput("monthly_charges_dist"),
    plotlyOutput("churn_bar")
  ),
  
  # Feature Importance Section
  div(
    id = "feature-importance",
    class = "section",
    h2("Feature Importance"),
    plotlyOutput("feature_importance")
  ),
  
  # ROC Curve Section
  div(
    id = "roc-curve",
    class = "section",
    h2("ROC Curve"),
    plotlyOutput("roc_curve")
  )
)


# Server logic remains unchanged
server <- function(input, output) {
  models <- reactiveValues(logistic = NULL, rf = NULL, logistic_pred_prob = NULL, rf_pred_prob = NULL)
  
  observeEvent(input$train_models, {
    models$logistic <- glm(Churn ~ ., data = train_data, family = binomial)
    models$rf <- randomForest(Churn ~ ., data = train_data, ntree = 100, mtry = 2)
    models$logistic_pred_prob <- predict(models$logistic, test_data, type = "response")
    models$rf_pred_prob <- predict(models$rf, test_data, type = "prob")[, 2]
    output$model_summary <- renderPrint({
      list(Logistic_Model = summary(models$logistic), Random_Forest_Model = models$rf)
    })
  })
  
  output$summary <- renderPrint({
    list(Structure = str(telco_data), Summary = summary(telco_data), Missing_Values = colSums(is.na(telco_data)))
  })
  
  observeEvent(input$predict, {
    if (is.null(models$logistic) || is.null(models$rf)) {
      output$predictions <- renderPrint({
        "Please train the models before making predictions."
      })
      return()
    }
    prediction_data <- data.frame(
      tenure = as.numeric(input$tenure),
      MonthlyCharges = as.numeric(input$MonthlyCharges),
      Contract = factor(input$Contract, levels = levels(train_data$Contract)),
      PaymentMethod = factor(input$PaymentMethod, levels = levels(train_data$PaymentMethod)),
      InternetService = factor(input$InternetService, levels = levels(train_data$InternetService))
    )
    logistic_pred <- predict(models$logistic, prediction_data, type = "response")
    logistic_class <- ifelse(logistic_pred > 0.5, "Yes", "No")
    rf_pred <- predict(models$rf, prediction_data)
    output$predictions <- renderPrint({
      list(Logistic_Prediction = logistic_class, Random_Forest_Prediction = rf_pred)
    })
  })
  
  output$tenure_dist <- renderPlotly({
    ggplotly(ggplot(telco_data, aes(x = tenure, fill = Churn)) + geom_histogram(bins = 30, position = "dodge") + theme_minimal())
  })
  
  output$monthly_charges_dist <- renderPlotly({
    ggplotly(ggplot(telco_data, aes(x = MonthlyCharges, fill = Churn)) + geom_histogram(bins = 30, position = "dodge") + theme_minimal())
  })
  
  output$churn_bar <- renderPlotly({
    ggplotly(ggplot(telco_data, aes(x = Churn)) + geom_bar(fill = c("salmon", "skyblue")) + theme_minimal())
  })
  
  output$feature_importance <- renderPlotly({
    if (!is.null(models$rf)) {
      rf_importance <- as.data.frame(importance(models$rf))
      rf_importance$Feature <- rownames(rf_importance)
      ggplotly(ggplot(rf_importance, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) + geom_bar(stat = "identity", fill = "steelblue") + coord_flip() + theme_minimal())
    }
  })
  
  output$roc_curve <- renderPlotly({
    if (!is.null(models$logistic_pred_prob) && !is.null(models$rf_pred_prob)) {
      logistic_roc <- roc(test_data$Churn, models$logistic_pred_prob)
      rf_roc <- roc(test_data$Churn, models$rf_pred_prob)
      logistic_roc_df <- data.frame(FPR = logistic_roc$specificities, TPR = logistic_roc$sensitivities, Model = "Logistic Regression")
      rf_roc_df <- data.frame(FPR = rf_roc$specificities, TPR = rf_roc$sensitivities, Model = "Random Forest")
      roc_data <- rbind(logistic_roc_df, rf_roc_df)
      ggplotly(ggplot(roc_data, aes(x = 1 - FPR, y = TPR, color = Model)) + geom_line(size = 1) + theme_minimal() + labs(title = "ROC Curve"))
    }
  })
}

shinyApp(ui = ui, server = server)
