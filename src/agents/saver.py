import os
from datetime import datetime
import pdfkit
from src.core.state import AgentState

def saver_node(state: AgentState):
    """
    Saves the report as a PDF by converting HTML. 
    This is the ONLY way to get perfect Arabic without manual font files.
    """
    report_content = state.get("draft", "")
    task_name = state.get("task", "Mantiq-AI Report")
    
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/report_{timestamp}.pdf"
    
    print(f"--- LOG: Mantiq-AI (Saver) is generating a Web-Standard PDF ---")

    # HTML Template with RTL support for perfect Arabic
    html_template = f"""
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; line-height: 1.6; }}
            h1 {{ color: #1a5f7a; text-align: center; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            pre {{ white-space: pre-wrap; font-family: inherit; font-size: 14px; }}
        </style>
    </head>
    <body>
        <h1>{task_name}</h1>
        <pre>{report_content}</pre>
    </body>
    </html>
    """

    try:
        # Options to handle the conversion
        options = {
            'encoding': "UTF-8",
            'quiet': ''
        }
        
        # Convert string to PDF
        pdfkit.from_string(html_template, filename, options=options)
        
        return {"next_step": "FINISH"}

    except Exception as e:
        print(f"--- ERROR: PDF Generation failed: {str(e)} ---")
        # If pdfkit fails, we fallback to Markdown so you don't lose the data
        fallback_file = filename.replace(".pdf", ".md")
        with open(fallback_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"--- INFO: Saved as Markdown instead: {fallback_file} ---")
        return {"next_step": "FINISH"}