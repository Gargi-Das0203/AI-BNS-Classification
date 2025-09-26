"""
BNS Section Classifier - Streamlit Web App
==========================================

Interactive web interface for classifying police complaints into BNS sections.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="BNS Section Classifier",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Auto Dark/Light Theme Support
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: var(--text-color);
        margin-bottom: 2rem;
    }
    
    /* Light theme styles */
    [data-theme="light"] {
        --bg-color: #ffffff;
        --card-bg: #f8f9fa;
        --text-color: #212529;
        --text-secondary: #6c757d;
        --border-color: #dee2e6;
    }
    
    /* Dark theme styles */
    [data-theme="dark"], .stApp {
        --bg-color: #0e1117;
        --card-bg: #262730;
        --text-color: #fafafa;
        --text-secondary: #a6a6a6;
        --border-color: #464853;
    }
    
    .section-card {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        border: 1px solid var(--border-color);
    }
    
    .section-card h4 {
        color: #4CAF50 !important;
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
    }
    
    .section-card h5 {
        color: var(--text-color) !important;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .section-card p {
        color: var(--text-secondary) !important;
        margin-bottom: 0.8rem;
        line-height: 1.5;
    }
    
    .section-card strong {
        color: var(--text-color) !important;
    }
    
    .confidence-high {
        color: #28a745 !important;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107 !important;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545 !important;
        font-weight: bold;
    }
    
    .footer-box {
        background: var(--card-bg) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
    }
    
    .footer-box strong {
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_bns_data():
    """Load BNS sections and training data."""
    try:
        bns_sections = pd.read_csv('backend/bns_sections.csv')
        train_data = pd.read_csv('backend/train_data.csv')
        return bns_sections, train_data, None
    except Exception as e:
        return None, None, str(e)

def keyword_classify(text, bns_sections):
    """Enhanced keyword-based classification."""
    text_lower = text.lower()
    
    # Enhanced keyword mapping with more comprehensive patterns
    keyword_patterns = {
        # Murder and Violence
        'murder|killed|death|assassination|homicide': [1, 2],
        'group.*murder|caste.*murder|communal.*murder': [2],
        
        # Theft and Property Crimes  
        'theft|stolen|stole|burglar|loot': [8, 10, 11, 12],
        'house.*theft|dwelling.*theft|broke.*house': [10],
        'snatching|snatch|grabbed': [9],
        
        # Robbery
        'robbery|robbed|dacoity|armed.*theft': [14, 15, 16, 17],
        
        # Sexual Offences
        'rape|sexual.*assault|molest': [22, 23, 24, 25, 26],
        'child.*rape|minor.*rape|under.*12': [23, 25],
        'gang.*rape|multiple.*men': [24, 26],
        
        # Public Order
        'riot|unlawful.*assembly|mob|crowd': [3, 4, 5, 6, 7],
        'deadly.*weapon|armed.*group': [7],
        
        # Trespass
        'trespass|entered.*house|unauthorized.*entry': [19, 20, 21],
        'lurking|hiding.*house': [21]
    }
    
    results = []
    confidence_scores = {}
    
    # Pattern matching with confidence scoring
    for pattern, section_ids in keyword_patterns.items():
        if re.search(pattern, text_lower):
            for section_id in section_ids:
                section = bns_sections[bns_sections['section_id'] == section_id]
                if not section.empty:
                    section_info = section.iloc[0]
                    
                    # Calculate confidence based on keyword match strength
                    base_confidence = 0.6
                    if len(re.findall(pattern, text_lower)) > 1:
                        base_confidence += 0.1
                    if len(text.split()) > 20:  # Longer text = more context
                        base_confidence += 0.1
                    
                    key = section_info['section_code']
                    if key not in confidence_scores or confidence_scores[key] < base_confidence:
                        confidence_scores[key] = min(base_confidence, 0.9)
                        
                        results.append({
                            'section_id': section_id,
                            'section_code': section_info['section_code'],
                            'section_title': section_info['section_title'],
                            'section_description': section_info['section_description'],
                            'crime_category': section_info['crime_category'],
                            'confidence': confidence_scores[key],
                            'method': 'Keyword Analysis'
                        })
    
    # Sort by confidence and return top 5
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results[:5]

def display_classification_results(results):
    """Display classification results in an attractive format."""
    if not results:
        st.warning("⚠️ No matching BNS sections found. Please try rephrasing your complaint or add more details.")
        return
    
    st.success(f"✅ Found {len(results)} matching BNS section(s)")
    
    for i, result in enumerate(results, 1):
        confidence = result['confidence']
        
        # Determine confidence level styling
        if confidence >= 0.7:
            conf_class = "confidence-high"
            conf_icon = "🟢"
        elif confidence >= 0.5:
            conf_class = "confidence-medium"  
            conf_icon = "🟡"
        else:
            conf_class = "confidence-low"
            conf_icon = "🔴"
        
        # Create result card
        with st.container():
            st.markdown(f"""
            <div class="section-card">
                <h4>#{i} {result['section_code']} {conf_icon}</h4>
                <h5>{result['section_title']}</h5>
                <p><strong>Category:</strong> {result['crime_category']}</p>
                <p><strong>Description:</strong> {result['section_description']}</p>
                <p><strong>Confidence:</strong> <span class="{conf_class}">{confidence:.1%}</span></p>
                <p><strong>Method:</strong> {result['method']}</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Initialize session state
    if 'complaint_text' not in st.session_state:
        st.session_state.complaint_text = ""
    if 'clear_clicked' not in st.session_state:
        st.session_state.clear_clicked = False
    
    # Header
    st.markdown('<div class="main-header">⚖️ BNS Section Classifier</div>', unsafe_allow_html=True)
    st.markdown("### Intelligent Classification of Police Complaints into Bharatiya Nyaya Sanhita (BNS) Sections")
    
    # Load data
    with st.spinner("Loading BNS data..."):
        bns_sections, train_data, error = load_bns_data()
    
    if error:
        st.error(f"❌ Error loading data: {error}")
        st.info("Please ensure the CSV files are in the 'backend/' directory.")
        return
    
    # Sidebar - System Information
    with st.sidebar:
        st.header("📊 System Information")
        st.info(f"**BNS Sections:** {len(bns_sections)}")
        st.info(f"**Training Samples:** {len(train_data)}")
        
        st.header("📋 Crime Categories")
        categories = bns_sections['crime_category'].value_counts()
        for category, count in categories.items():
            st.write(f"• **{category}:** {count} sections")
        
        st.header("🔍 Sample BNS Sections")
        sample_sections = bns_sections.head(5)
        for _, section in sample_sections.iterrows():
            st.write(f"• {section['section_code']}")
        
        st.header("ℹ️ How to Use")
        st.write("""
        1. Enter a police complaint in the text area
        2. Click 'Classify Complaint'
        3. Review the suggested BNS sections
        4. Each result shows confidence level and reasoning
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Police Complaint")
        
        # Sample complaints for quick testing
        sample_complaints = {
            "Select a sample...": "",
            "Caste-motivated Murder": "A group of 7 men from upper caste community brutally murdered Mr. Rajesh, a Dalit teacher, while shouting casteist slurs and saying 'people like you should know your place'.",
            "House Burglary": "Three masked thieves broke into my house at 2 AM by picking the door lock and stole jewelry worth Rs. 5 lakhs, cash, and electronics.",
            "Child Sexual Assault": "A 45-year-old man lured my 10-year-old daughter from the park and sexually assaulted her. She is now hospitalized with severe injuries.",
            "Mobile Snatching": "Two men on a motorcycle snatched my phone while I was walking on the road. They pushed me down and drove away quickly.",
            "Rioting with Weapons": "A mob of 50 people armed with sticks and stones attacked our community center during the festival, breaking windows and threatening residents."
        }
        
        selected_sample = st.selectbox("Quick Test - Select a sample complaint:", list(sample_complaints.keys()))
        
        # Initialize text area key in session state
        if 'text_area_key' not in st.session_state:
            st.session_state.text_area_key = 0
        
        # Determine default text based on selection
        if selected_sample != "Select a sample...":
            default_text = sample_complaints.get(selected_sample, "")
        else:
            default_text = ""
        
        complaint_text = st.text_area(
            "Enter complaint details:",
            value=default_text,
            height=450,
            placeholder="Describe the incident in detail including what happened, when, where, and who was involved...",
            key=f"complaint_text_{st.session_state.text_area_key}",
            help="💡 Tip: Provide as much detail as possible for better BNS section classification"
        )
        
        col_classify, col_clear = st.columns([1, 1])
        
        with col_classify:
            classify_button = st.button("🔍 Classify Complaint", type="primary", use_container_width=True)
        
        with col_clear:
            if st.button("🗑️ Clear Text", use_container_width=True):
                # Increment key to force text area to reset
                st.session_state.text_area_key += 1
                st.rerun()
    
    with col2:
        st.header("📈 Statistics")
        
        if bns_sections is not None:
            # Show distribution chart
            fig_data = bns_sections['crime_category'].value_counts()
            st.bar_chart(fig_data)
            
            st.subheader("🎯 Classification Tips")
            st.info("""
            **For better results:**
            • Include specific details (date, time, location)
            • Mention weapons or tools used
            • Describe the relationship between parties
            • Include victim demographics if relevant
            • Specify the nature of harm or loss
            """)
    
    # Classification Results
    if classify_button and complaint_text.strip():
        st.markdown("---")
        st.header("🎯 Classification Results")
        
        with st.spinner("Analyzing complaint and identifying applicable BNS sections..."):
            results = keyword_classify(complaint_text, bns_sections)
        
        display_classification_results(results)
        
        # Additional Information
        if results:
            st.markdown("---")
            st.header("📚 Legal Context")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚖️ Suggested Actions")
                st.write("""
                1. **File FIR** at the nearest police station
                2. **Gather evidence** (photos, witnesses, documents)
                3. **Consult legal counsel** for complex cases
                4. **Follow up** on investigation progress
                """)
            
            with col2:
                st.subheader("📞 Emergency Contacts")
                st.write("""
                • **Police Emergency:** 100
                • **Women Helpline:** 1091
                • **Child Helpline:** 1098
                • **Cyber Crime:** 1930
                """)
    
    elif classify_button:
        st.warning("⚠️ Please enter a complaint description to classify.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-box">
        <p style="margin-bottom: 1rem;">🔒 <strong>Disclaimer:</strong> This tool provides automated suggestions based on keyword analysis. 
        Always consult with legal professionals for official legal advice.</p>
        <p>⚖️ <strong>BNS Section Classifier</strong> | Built with Streamlit | September 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()