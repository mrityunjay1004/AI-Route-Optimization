# AI-Route-Optimization
AI Route Optimization System

This project is an AI-powered logistics optimization system that helps improve delivery efficiency by finding the most optimal route between multiple locations. It combines route optimization algorithms, basic machine learning for ETA prediction, and interactive map visualization.

Users can either upload a dataset or manually enter delivery details such as pickup city, delivery city, package weight, traffic conditions, and cost parameters. The system automatically converts city names into geographic coordinates and computes the most efficient route using optimization techniques.

The application also provides real-time business insights by comparing a naive route with the optimized route. It calculates key performance metrics such as total distance, fuel consumption, delivery time, and profit improvement, helping logistics operations become more efficient and cost-effective.

Key Features
Route optimization using Vehicle Routing Problem (VRP)
ETA prediction using machine learning
City-to-coordinate conversion (geocoding)
Interactive map visualization
Manual and dataset-based input
Business insights: fuel saved, time saved, profit increase
Tech Stack
Python
Streamlit (Web App)
OR-Tools (Optimization)
Scikit-learn (ML)
Folium (Maps)
Geopy (Geocoding)
