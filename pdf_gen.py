import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors



def generate_pdf_report(forecast, metrics, forecast_image_path, yoy_image_path, cleaned_data=None, fe_data=None, date_col=None, target_col=None):
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph(f"Forecast Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Summary Metrics
    summary_text = f"""
    <b>Key Metrics:</b><br/>
    - Total Forecast Value: {metrics['total_forecast']:,.2f}<br/>
    - Latest Forecast Value: {metrics['latest_forecast']:,.2f}<br/>
    - Average YoY Growth: {metrics['average_yoy']:.2f}%
    """
    elements.append(Paragraph(summary_text, styles['BodyText']))
    elements.append(Spacer(1, 12))

    # Forecast Graph
    elements.append(Paragraph("<b>Forecast Graph:</b>", styles['Heading2']))
    elements.append(Image(forecast_image_path, width=400, height=200))
    elements.append(Spacer(1, 12))

    # YoY Growth Graph
    elements.append(Paragraph("<b>Year-over-Year Growth:</b>", styles['Heading2']))
    elements.append(Image(yoy_image_path, width=400, height=200))
    elements.append(Spacer(1, 12))

    # Forecast Table
    elements.append(Paragraph("<b>Forecast Table:</b>", styles['Heading2']))
    table_data = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(10).values.tolist()
    table_data.insert(0, ["Date", "Forecast Value", "Lower Bound", "Upper Bound"])
    table = Table(table_data, colWidths=[100, 100, 100, 100])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(table)

    pdf.build(elements)
    buffer.seek(0)
    return buffer