import pypdf
reader = pypdf.PdfReader('CV_Course_Project_Traffic_Violation - Copy.pdf')
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"
print(text)
