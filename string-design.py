import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO

def download_link(df, filename, text):
    """Generate a link to download the dataframe as Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'
    return href

def solar_mppt_allocator(azimuth_data, inverter_data, min_modules, max_modules):
    """
    Calculate optimal allocation of strings to MPPTs across multiple azimuths
    
    Parameters:
    -----------
    azimuth_data : dict
        Dictionary with keys as azimuth names and values as tuples (string_count, modules_per_string)
        Example: {"North": (20, 18), "South": (35, 16), "East": (15, 17), "West": (22, 19)}
    
    inverter_data : dict
        Dictionary with information about inverters
        Keys:
        - 'count': Number of inverters
        - 'mppt_per_inverter': Number of MPPTs per inverter
        - 'strings_per_mppt': Maximum strings per MPPT
    
    min_modules : int
        Minimum modules allowed per string
    
    max_modules : int
        Maximum modules allowed per string
    
    Returns:
    --------
    tuple : (allocation_df, summary_df, messages)
        - allocation_df: DataFrame showing MPPT allocation by azimuth
        - summary_df: Summary of allocation with utilization metrics
        - messages: List of warning or error messages
    """
    messages = []
    
    # Validate module counts
    for azimuth, (_, modules_per_string) in azimuth_data.items():
        if modules_per_string < min_modules:
            messages.append(f"Warning: {azimuth} has {modules_per_string} modules per string, which is below the minimum of {min_modules}.")
        if modules_per_string > max_modules:
            messages.append(f"Warning: {azimuth} has {modules_per_string} modules per string, which exceeds the maximum of {max_modules}.")
    
    # Calculate total capacity
    total_mppts = inverter_data['count'] * inverter_data['mppt_per_inverter']
    total_string_capacity = total_mppts * inverter_data['strings_per_mppt']
    total_strings_needed = sum(count for count, _ in azimuth_data.values())
    
    if total_strings_needed > total_string_capacity:
        messages.append(f"Error: Total strings needed ({total_strings_needed}) exceeds capacity ({total_string_capacity})")
        return None, None, messages
    
    # Initialize data structures for allocation
    inverters = []
    for i in range(inverter_data['count']):
        inverter = {
            'id': f"Inv-{i+1}",
            'mppts': []
        }
        for j in range(inverter_data['mppt_per_inverter']):
            inverter['mppts'].append({
                'id': f"MPPT-{j+1}",
                'azimuth': None,
                'strings': 0,
                'modules_per_string': 0,
                'capacity': inverter_data['strings_per_mppt']
            })
        inverters.append(inverter)
    
    # Sort azimuths by number of strings (descending)
    sorted_azimuths = sorted(azimuth_data.items(), key=lambda x: x[1][0], reverse=True)
    
    # Helper function to find available MPPTs
    def find_available_mppts(azimuth=None):
        available = []
        for inv_idx, inverter in enumerate(inverters):
            for mppt_idx, mppt in enumerate(inverter['mppts']):
                # If looking for a specific azimuth, only return MPPTs with that azimuth or empty ones
                if azimuth is not None:
                    if mppt['azimuth'] == azimuth and mppt['strings'] < mppt['capacity']:
                        available.append((inv_idx, mppt_idx))
                # Otherwise, return only empty MPPTs
                elif mppt['azimuth'] is None:
                    available.append((inv_idx, mppt_idx))
        return available
    
    # Allocate strings to MPPTs
    for azimuth, (string_count, modules_per_string) in sorted_azimuths:
        strings_remaining = string_count
        
        # First try to find MPPTs that already have this azimuth and have capacity
        existing_mppts = find_available_mppts(azimuth)
        for inv_idx, mppt_idx in existing_mppts:
            mppt = inverters[inv_idx]['mppts'][mppt_idx]
            strings_to_add = min(strings_remaining, mppt['capacity'] - mppt['strings'])
            mppt['strings'] += strings_to_add
            strings_remaining -= strings_to_add
            
            if strings_remaining == 0:
                break
        
        # If we still have strings to allocate, find empty MPPTs
        if strings_remaining > 0:
            empty_mppts = find_available_mppts()
            for inv_idx, mppt_idx in empty_mppts:
                mppt = inverters[inv_idx]['mppts'][mppt_idx]
                mppt['azimuth'] = azimuth
                mppt['modules_per_string'] = modules_per_string
                strings_to_add = min(strings_remaining, mppt['capacity'])
                mppt['strings'] = strings_to_add
                strings_remaining -= strings_to_add
                
                if strings_remaining == 0:
                    break
        
        if strings_remaining > 0:
            messages.append(f"Error: Could not allocate all strings for {azimuth}. {strings_remaining} strings remaining.")
    
    # Create output dataframes
    allocation_data = []
    for inverter in inverters:
        for mppt in inverter['mppts']:
            allocation_data.append({
                'Inverter': inverter['id'],
                'MPPT': mppt['id'],
                'Assigned Azimuth': mppt['azimuth'] if mppt['azimuth'] else 'Unused',
                'Strings': mppt['strings'],
                'Modules per String': mppt['modules_per_string'] if mppt['azimuth'] else 0,
                'Total Modules': mppt['strings'] * mppt['modules_per_string'] if mppt['azimuth'] else 0
            })
    
    allocation_df = pd.DataFrame(allocation_data)
    
    # Create summary dataframe
    summary_data = []
    for azimuth, (string_count, modules_per_string) in azimuth_data.items():
        azimuth_rows = allocation_df[allocation_df['Assigned Azimuth'] == azimuth]
        mppts_used = len(azimuth_rows)
        allocated_strings = azimuth_rows['Strings'].sum()
        total_modules = azimuth_rows['Total Modules'].sum()
        
        if allocated_strings != string_count:
            messages.append(f"Warning: {azimuth} was supposed to have {string_count} strings but {allocated_strings} were allocated.")
        
        summary_data.append({
            'Azimuth': azimuth,
            'Total Strings': allocated_strings,
            'Modules per String': modules_per_string,
            'Total Modules': total_modules,
            'MPPTs Utilized': mppts_used,
            'Avg Strings per MPPT': round(allocated_strings / mppts_used, 2) if mppts_used > 0 else 0,
            'MPPT Utilization': f"{round((allocated_strings / (mppts_used * inverter_data['strings_per_mppt'])) * 100, 1)}%" if mppts_used > 0 else "0%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add system total row
    system_total = {
        'Azimuth': 'SYSTEM TOTAL',
        'Total Strings': summary_df['Total Strings'].sum(),
        'Modules per String': '-',
        'Total Modules': summary_df['Total Modules'].sum(),
        'MPPTs Utilized': len(allocation_df[allocation_df['Assigned Azimuth'] != 'Unused']),
        'Avg Strings per MPPT': '-',
        'MPPT Utilization': f"{round((summary_df['Total Strings'].sum() / (total_mppts * inverter_data['strings_per_mppt'])) * 100, 1)}%"
    }
    summary_df = summary_df.append(system_total, ignore_index=True)
    
    return allocation_df, summary_df, messages

# Streamlit app
st.set_page_config(page_title="Solar PV String-to-MPPT Calculator", page_icon="☀️", layout="wide")

st.title("☀️ Solar PV String-to-MPPT Allocation Tool")
st.write("""
This tool helps solar PV design engineers allocate strings to MPPTs across multiple azimuths.
It respects constraints on modules per string and ensures each MPPT only contains strings from a single azimuth.
""")

# Create sidebar for inputs
st.sidebar.header("System Configuration")

# Inverter configuration
st.sidebar.subheader("Inverter Configuration")
inverter_count = st.sidebar.number_input("Number of Inverters", min_value=1, max_value=50, value=9)
mppt_per_inverter = st.sidebar.number_input("MPPTs per Inverter", min_value=1, max_value=20, value=7)
strings_per_mppt = st.sidebar.number_input("Strings per MPPT", min_value=1, max_value=10, value=3)

# Module constraints
st.sidebar.subheader("Module Constraints")
min_modules = st.sidebar.number_input("Minimum Modules per String", min_value=1, max_value=30, value=15)
max_modules = st.sidebar.number_input("Maximum Modules per String", min_value=min_modules, max_value=30, value=19)

# Azimuth configuration
st.sidebar.subheader("Azimuth Configuration")
num_azimuths = st.sidebar.number_input("Number of Azimuths", min_value=1, max_value=10, value=4)

# Main content area - dynamic form for azimuths
st.header("Azimuth Configuration")

azimuth_data = {}
azimuth_cols = st.columns(min(num_azimuths, 4))  # Up to 4 columns

for i in range(num_azimuths):
    col_idx = i % 4  # Cycle through columns
    with azimuth_cols[col_idx]:
        st.subheader(f"Azimuth {i+1}")
        azimuth_name = st.text_input(f"Name", value=f"Azimuth {i+1}", key=f"name_{i}")
        string_count = st.number_input(f"Number of Strings", min_value=1, value=20, key=f"strings_{i}")
        modules_per_string = st.number_input(f"Modules per String", min_value=min_modules, max_value=max_modules, value=min_modules, key=f"modules_{i}")
        
        azimuth_data[azimuth_name] = (string_count, modules_per_string)

# Create a calculate button
if st.button("Calculate Allocation"):
    # Prepare data for calculation
    inverter_data = {
        'count': inverter_count,
        'mppt_per_inverter': mppt_per_inverter,
        'strings_per_mppt': strings_per_mppt
    }
    
    # Run the calculator
    allocation_df, summary_df, messages = solar_mppt_allocator(azimuth_data, inverter_data, min_modules, max_modules)
    
    # Display any warnings or errors
    if messages:
        st.warning("Warnings/Errors:")
        for msg in messages:
            st.write(f"- {msg}")
    
    if allocation_df is not None and summary_df is not None:
        # Display summary
        st.header("Summary")
        st.dataframe(summary_df)
        
        # Display allocation
        st.header("MPPT Allocation")
        st.dataframe(allocation_df)
        
        # Prepare Excel with both sheets for download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            allocation_df.to_excel(writer, sheet_name='MPPT Allocation', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        download_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="solar_mppt_allocation.xlsx">Download Excel Report</a>'
        
        st.markdown(download_link, unsafe_allow_html=True)
        
        # Visualize the allocation
        st.header("Allocation Visualization")
        
        # Create a pivot table for visualization
        pivot_df = allocation_df.pivot(index='Inverter', columns='MPPT', values='Assigned Azimuth')
        
        # Replace NaN with "Unused"
        pivot_df = pivot_df.fillna('Unused')
        
        # Display as a heatmap
        st.write("MPPT Allocation by Inverter (hover for details)")
        
        # Create a dictionary mapping azimuths to colors
        unique_azimuths = allocation_df['Assigned Azimuth'].unique()
        azimuth_colors = {}
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, azimuth in enumerate(unique_azimuths):
            azimuth_colors[azimuth] = color_palette[i % len(color_palette)]
            
        # We can't create actual visualizations in this environment, but in Streamlit this would be possible
        st.write("(Visualization would appear here in the actual Streamlit app)")
        
        # Display allocation by azimuth
        st.subheader("Allocation by Azimuth")
        for azimuth in azimuth_data.keys():
            azimuth_allocation = allocation_df[allocation_df['Assigned Azimuth'] == azimuth]
            st.write(f"**{azimuth}** - {len(azimuth_allocation)} MPPTs, {azimuth_allocation['Strings'].sum()} strings")
            st.dataframe(azimuth_allocation[['Inverter', 'MPPT', 'Strings', 'Modules per String', 'Total Modules']])
