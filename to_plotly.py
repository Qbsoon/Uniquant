import json
import os
import argparse
import plotly.graph_objects as go
import tempfile
import time

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

def generate_plot(json_files, y_param, labels, output_name=None):
    if len(json_files) != len(labels):
        print("Błąd: Liczba plików JSON musi być równa liczbie etykiet.")
        return

    fig = go.Figure()
    all_x_labels = set()

    for file_path, label in zip(json_files, labels):
        if not os.path.exists(file_path):
            print(f"Ostrzeżenie: Plik {file_path} nie istnieje. Pomijanie.")
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        points = []
        for entry in data:
            if 'quant_size' not in entry:
                x_label = "original"
                sort_key = (0, 0)
            else:
                q = int(entry['quant_size'])
                p = int(entry['pack_size'])
                x_label = f"{q}-{p}"
                sort_key = (q, p)
            
            y_val = entry.get(y_param)
            
            if y_val is not None:
                points.append({
                    'x': x_label,
                    'y': y_val,
                    'key': sort_key
                })
                all_x_labels.add(x_label)

        points.sort(key=lambda x: x['key'])
        
        x_vals = [p['x'] for p in points]
        y_vals = [p['y'] for p in points]
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            name=label
        ))

    def get_sort_key(label):
        if label == "original":
            return (0, 0)
        parts = label.split('-')
        return (int(parts[0]), int(parts[1]))

    sorted_categories = sorted(list(all_x_labels), key=get_sort_key)

    fig.update_layout(
        title=f"Wykres parametru: {y_param}",
        xaxis_title="Konfiguracja (Quant Size - Pack Size)",
        yaxis_title=y_param,
        legend_title="Źródło danych",
        hovermode="x unified",
        width=1920, 
        height=1080,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    fig.update_xaxes(
        type='category',
        categoryorder='array',
        categoryarray=sorted_categories
    )
    
    if output_name is None:
        base_name = os.path.splitext(json_files[0])[0]
        output_filename = f"{base_name}.png"
    else:
        output_filename = output_name

    try:
        print("Renderowanie wykresu przez Firefox...")
        
        options = Options()
        options.add_argument("--headless")
        
        driver = webdriver.Firefox(options=options)
        driver.set_window_size(2000, 1200)
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            temp_html = f.name
            fig.write_html(temp_html, include_plotlyjs='cdn', config={'displayModeBar': False})
        
        driver.get(f"file://{temp_html}")
        time.sleep(3)
        
        plotly_element = driver.find_element(By.CLASS_NAME, "plotly-graph-div")
        png_data = plotly_element.screenshot_as_png
        
        with open(output_filename, 'wb') as f:
            f.write(png_data)
            
        driver.quit()
        os.remove(temp_html)
        
        print(f"Pomyślnie zapisano: {output_filename}")
        
    except Exception as e:
        print(f"Błąd renderowania: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profesjonalny eksport wykresów (Firefox)")
    parser.add_argument("param", help="Parametr na oś Y")
    parser.add_argument("files", nargs='+', help="Pliki JSON")
    parser.add_argument("--labels", nargs='+', required=True, help="Etykiety")
    parser.add_argument("--output", help="Plik wyjściowy")
    
    args = parser.parse_args()
    generate_plot(args.files, args.param, args.labels, args.output)