"""
Gradio web application for house price prediction.
Features 3 visualizations: feature importance, input summary, and prediction with confidence interval.
"""

import gradio as gr
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load model artifacts
print("Loading model artifacts...")
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def create_feature_importance_plot():
    """
    Visualization 1: Feature importance bar chart.
    Shows which features contribute most to predictions.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance in House Price Prediction', fontsize=16, fontweight='bold')
    plt.bar(range(len(importances)), importances[indices], color='steelblue', alpha=0.8)
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.tight_layout()
    
    plot_path = 'plots/feature_importance.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_input_summary_plot(square_feet, bedrooms, bathrooms, age_years, 
                               lot_size, garage_spaces, neighborhood_score):
    """
    Visualization 2: Input values summary radar/bar chart.
    Displays the current input values in a visual format.
    """
    # Normalize inputs to 0-1 scale for visualization
    inputs = {
        'Square Feet': square_feet / 4000,
        'Bedrooms': bedrooms / 5,
        'Bathrooms': bathrooms / 4,
        'Age (Years)': age_years / 50,
        'Lot Size': lot_size / 10000,
        'Garage': garage_spaces / 3,
        'Neighborhood': neighborhood_score / 10
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(inputs)))
    ax1.barh(list(inputs.keys()), list(inputs.values()), color=colors, alpha=0.8)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Normalized Value (0-1)', fontsize=11)
    ax1.set_title('Input Features (Normalized)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Actual values table
    actual_values = [
        f'{square_feet:,.0f} sq ft',
        f'{bedrooms:.0f}',
        f'{bathrooms:.0f}',
        f'{age_years:.0f} yrs',
        f'{lot_size:,.0f} sq ft',
        f'{garage_spaces:.0f}',
        f'{neighborhood_score:.1f}/10'
    ]
    
    ax2.axis('tight')
    ax2.axis('off')
    table_data = [[k, v] for k, v in zip(inputs.keys(), actual_values)]
    table = ax2.table(cellText=table_data, colLabels=['Feature', 'Value'],
                     cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax2.set_title('Actual Input Values', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plot_path = 'plots/input_summary.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_prediction_plot(predicted_price, confidence_interval):
    """
    Visualization 3: Prediction result with 95% confidence interval.
    Shows predicted price and confidence bounds.
    """
    lower_bound = predicted_price - confidence_interval
    upper_bound = predicted_price + confidence_interval
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main prediction bar
    ax.barh(['Predicted Price'], [predicted_price], height=0.4, 
            color='#2E7D32', alpha=0.8, label='Prediction')
    
    # Confidence interval
    ax.barh(['Predicted Price'], [upper_bound - lower_bound], 
            left=lower_bound, height=0.6, 
            color='lightblue', alpha=0.4, label='95% Confidence Interval')
    
    # Add value labels
    ax.text(predicted_price, 0, f'${predicted_price:,.0f}', 
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(lower_bound, 0, f'${lower_bound:,.0f}', 
            ha='right', va='top', fontsize=10, style='italic')
    ax.text(upper_bound, 0, f'${upper_bound:,.0f}', 
            ha='left', va='top', fontsize=10, style='italic')
    
    ax.set_xlabel('Price ($)', fontsize=12)
    ax.set_title('House Price Prediction with 95% Confidence Interval', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = 'plots/prediction.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return plot_path

def predict_price(square_feet, bedrooms, bathrooms, age_years, 
                  lot_size, garage_spaces, neighborhood_score):
    """
    Main prediction function that generates all visualizations.
    
    Returns:
        tuple: (prediction_text, feature_importance_plot, input_summary_plot, prediction_plot)
    """
    # Prepare input data
    input_data = np.array([[square_feet, bedrooms, bathrooms, age_years, 
                           lot_size, garage_spaces, neighborhood_score]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    predicted_price = model.predict(input_scaled)[0]
    
    # Calculate 95% confidence interval
    # Use predictions from all trees in the forest
    tree_predictions = np.array([tree.predict(input_scaled)[0] 
                                 for tree in model.estimators_])
    std_prediction = np.std(tree_predictions)
    confidence_interval = 1.96 * std_prediction  # 95% CI
    
    # Create visualizations
    feature_plot = create_feature_importance_plot()
    input_plot = create_input_summary_plot(square_feet, bedrooms, bathrooms, 
                                           age_years, lot_size, garage_spaces, 
                                           neighborhood_score)
    prediction_plot = create_prediction_plot(predicted_price, confidence_interval)
    
    # Format prediction text
    prediction_text = f"""
    ## Predicted House Price: ${predicted_price:,.2f}
    
    **95% Confidence Interval:** ${predicted_price - confidence_interval:,.2f} - ${predicted_price + confidence_interval:,.2f}
    
    This means we are 95% confident that the actual price falls within this range.
    """
    
    return prediction_text, feature_plot, input_plot, prediction_plot

# Create Gradio interface
with gr.Blocks(title="House Price Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏠 House Price Prediction System
    
    Enter the property details below to get an AI-powered price prediction with confidence intervals and visualizations.
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Property Features")
            
            square_feet = gr.Slider(
                minimum=800, maximum=4000, value=2000, step=50,
                label="Square Feet", info="Total living area"
            )
            
            bedrooms = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="Bedrooms", info="Number of bedrooms"
            )
            
            bathrooms = gr.Slider(
                minimum=1, maximum=4, value=2, step=1,
                label="Bathrooms", info="Number of bathrooms"
            )
            
            age_years = gr.Slider(
                minimum=0, maximum=50, value=10, step=1,
                label="Age (Years)", info="Age of the property"
            )
            
            lot_size = gr.Slider(
                minimum=2000, maximum=10000, value=5000, step=100,
                label="Lot Size (sq ft)", info="Total land area"
            )
            
            garage_spaces = gr.Slider(
                minimum=0, maximum=3, value=2, step=1,
                label="Garage Spaces", info="Number of parking spots"
            )
            
            neighborhood_score = gr.Slider(
                minimum=1, maximum=10, value=7, step=0.1,
                label="Neighborhood Score", info="Quality rating (1-10)"
            )
            
            predict_btn = gr.Button("Predict Price", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("### Prediction Results")
            prediction_output = gr.Markdown()
    
    gr.Markdown("---")
    gr.Markdown("### Visualizations")
    
    with gr.Row():
        feature_importance_plot = gr.Image(label="Feature Importance", type="filepath")
        input_summary_plot = gr.Image(label="Your Input Summary", type="filepath")
    
    with gr.Row():
        prediction_plot = gr.Image(label="Prediction with Confidence Interval", type="filepath")
    
    # Set up prediction on button click
    predict_btn.click(
        fn=predict_price,
        inputs=[square_feet, bedrooms, bathrooms, age_years, 
                lot_size, garage_spaces, neighborhood_score],
        outputs=[prediction_output, feature_importance_plot, 
                input_summary_plot, prediction_plot]
    )
    
    gr.Markdown("""
    ---
    ### How to interpret the results:
    
    - **Feature Importance**: Shows which property features have the biggest impact on price predictions
    - **Input Summary**: Visual representation of your entered values
    - **Prediction**: The estimated price with a 95% confidence interval (statistical uncertainty range)
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
