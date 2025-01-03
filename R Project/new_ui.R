# Load required libraries
library(shiny)
library(shinydashboard)
library(shinyjs)
library(readr)
library(dplyr)
library(caret)
library(randomForest)
library(ggplot2)
library(plotly)
library(pROC)

# Temporary user storage
users <- data.frame(
  username = c("admin"),
  password = c("password"),
  stringsAsFactors = FALSE
)

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

# Login UI
login_ui <- fluidPage(
  useShinyjs(),
  tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "styles.css")
  ),
  tags$div(
    class = "login-container",
    align = "center",
    tags$h2("Telco Churn Prediction Login"),
    textInput("login_username", "Username"),
    passwordInput("login_password", "Password"),
    actionButton("login_btn", "Login", class = "btn-primary"),
    actionButton("register_btn", "Register New Account", class = "btn-secondary"),
    uiOutput("login_message")
  )
)

# Registration UI
register_ui <- fluidPage(
  useShinyjs(),
  tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "styles.css")
  ),
  tags$div(
    class = "register-container",
    align = "center",
    tags$h2("Register New Account"),
    textInput("register_username", "Choose Username"),
    passwordInput("register_password", "Choose Password"),
    passwordInput("register_confirm_password", "Confirm Password"),
    actionButton("submit_register_btn", "Create Account", class = "btn-primary"),
    actionButton("back_to_login_btn", "Back to Login", class = "btn-secondary"),
    uiOutput("register_message")
  )
)


# Main Dashboard UI
dashboard_ui <- dashboardPage(
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
    tabItems(
      tabItem(tabName = "intro", h3("Welcome to the Telco Churn Prediction App!")),
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
          tabPanel("2D Visualizations", plotlyOutput("tenure_dist"), plotlyOutput("monthly_charges_dist")),
          tabPanel("3D Visualizations", plotlyOutput("scatter_3d"))
        )
      ),
      tabItem(tabName = "feature_importance", plotlyOutput("feature_importance")),
      tabItem(tabName = "roc_curve", plotlyOutput("roc_curve"))
    )
  )
)

# Server logic
server <- function(input, output, session) {
  # Reactive values for login state
  rv <- reactiveValues(is_logged_in = FALSE, page = "login")
  
  # Render UI based on login state
  output$ui <- renderUI({
    if (rv$page == "login") {
      login_ui
    } else if (rv$page == "register") {
      register_ui
    } else {
      dashboard_ui
    }
  })
  
  # Handle login
  observeEvent(input$login_btn, {
    if (any(users$username == input$login_username & users$password == input$login_password)) {
      rv$is_logged_in <- TRUE
      rv$page <- "dashboard"
    } else {
      showNotification("Invalid credentials!", type = "error")
    }
  })
  
  # Handle registration
  observeEvent(input$register_btn, { rv$page <- "register" })
  observeEvent(input$back_to_login_btn, { rv$page <- "login" })
  
  observeEvent(input$submit_register_btn, {
    if (input$register_password == input$register_confirm_password &&
        !(input$register_username %in% users$username)) {
      users <<- rbind(users, data.frame(username = input$register_username, password = input$register_password, stringsAsFactors = FALSE))
      showNotification("Account created successfully!", type = "message")
      rv$page <- "login"
    } else {
      showNotification("Registration failed. Check inputs.", type = "error")
    }
  })
  
  # Data summary
  output$data_summary <- renderPrint({
    list(Structure = capture.output(str(telco_data)), Summary = summary(telco_data))
  })
  
  # Train models
  models <- reactiveValues(logistic = NULL, rf = NULL)
  
  observeEvent(input$train_models, {
    models$logistic <- glm(Churn ~ ., data = train_data, family = binomial)
    models$rf <- randomForest(Churn ~ ., data = train_data, ntree = 100)
    output$model_summary <- renderPrint({
      list(Logistic = summary(models$logistic), Random_Forest = models$rf)
    })
  })
  
  # Prediction
  observeEvent(input$predict, {
    req(models$logistic, models$rf)
    input_data <- data.frame(
      tenure = input$tenure,
      MonthlyCharges = input$MonthlyCharges,
      Contract = factor(input$Contract, levels = levels(train_data$Contract)),
      PaymentMethod = factor(input$PaymentMethod, levels = levels(train_data$PaymentMethod)),
      InternetService = factor(input$InternetService, levels = levels(train_data$InternetService))
    )
    logistic_pred <- predict(models$logistic, input_data, type = "response")
    rf_pred <- predict(models$rf, input_data)
    output$predictions <- renderPrint({
      list(Logistic = ifelse(logistic_pred > 0.5, "Yes", "No"), Random_Forest = rf_pred)
    })
  })
  
  # Visualizations
  output$tenure_dist <- renderPlotly({
    ggplotly(ggplot(telco_data, aes(x = tenure, fill = Churn)) + geom_histogram(bins = 30) + theme_minimal())
  })
  
  output$monthly_charges_dist <- renderPlotly({
    ggplotly(ggplot(telco_data, aes(x = MonthlyCharges, fill = Churn)) + geom_histogram(bins = 30) + theme_minimal())
  })
  
  output$scatter_3d <- renderPlotly({
    plot_ly(telco_data, x = ~tenure, y = ~MonthlyCharges, z = ~as.numeric(Churn), color = ~Churn, type = "scatter3d", mode = "markers")
  })
  
  # Feature importance
  output$feature_importance <- renderPlotly({
    if (!is.null(models$rf)) {
      importance <- varImp(models$rf)
      ggplotly(ggplot(importance, aes(x = reorder(rownames(importance), Overall), y = Overall)) +
                 geom_bar(stat = "identity") +
                 coord_flip() +
                 theme_minimal())
    }
  })
  
  # ROC curve
  output$roc_curve <- renderPlotly({
    if (!is.null(models$logistic)) {
      logistic_probs <- predict(models$logistic, test_data, type = "response")
      rf_probs <- predict(models$rf, test_data, type = "prob")[, 2]
      
      logistic_roc <- roc(test_data$Churn, logistic_probs)
      rf_roc <- roc(test_data$Churn, rf_probs)
      
      plot_ly() %>%
        add_lines(x = 1 - logistic_roc$specificities, y = logistic_roc$sensitivities, name = "Logistic Regression") %>%
        add_lines(x = 1 - rf_roc$specificities, y = rf_roc$sensitivities, name = "Random Forest") %>%
        layout(title = "ROC Curve", xaxis = list(title = "1 - Specificity"), yaxis = list(title = "Sensitivity"))
    }
  })
}

# Run the app
shinyApp(ui = uiOutput("ui"), server = server)

