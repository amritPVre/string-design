import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import math

def download_link(df, filename, text):
    """Generate a link to download the dataframe as Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'
    return href

def calculate_optimal_string_configuration(total_modules, min_modules, max_modules, strings_per_mppt):
    """
    Calculate the optimal string configuration given total modules and constraints
    
    Returns a list of tuples (string_count, modules_per_string)
    """
    # Try all possible combinations of string sizes
    best_solution = None
    min_unused_modules = float('inf')
    
    # First, try to find a perfect solution with zero waste
    for size_distribution in range(1, 4):  # Try using 1, 2, or 3 different string sizes
        if size_distribution == 1:
            # Single string size
            for modules_per_string in range(min_modules, max_modules + 1):
                string_count = total_modules // modules_per_string
                unused_modules = total_modules % modules_per_string
                
                # If we found a perfect fit, return immediately
                if unused_modules == 0:
                    return [(string_count, modules_per_string)]
                
                # Otherwise, save if it's better than what we have
                if string_count > 0 and unused_modules < min_unused_modules:
                    min_unused_modules = unused_modules
                    best_solution = [(string_count, modules_per_string)]
        
        elif size_distribution == 2:
            # Two different string sizes
            for size1 in range(min_modules, max_modules + 1):
                for size2 in range(min_modules, max_modules + 1):
                    if size1 == size2:
                        continue
                    
                    # Try different combinations of size1 and size2
                    for count1 in range(1, (total_modules // size1) + 1):
                        remaining = total_modules - (count1 * size1)
                        count2 = remaining // size2
                        unused = remaining % size2
                        
                        if count2 <= 0:
                            continue
                            
                        if unused < min_unused_modules:
                            min_unused_modules = unused
                            best_solution = [(count1, size1), (count2, size2)]
                            
                            # If perfect solution, return immediately
                            if unused == 0:
                                return best_solution
        
        elif size_distribution == 3:
            # Three different string sizes (more computationally intensive)
            for size1 in range(min_modules, max_modules + 1):
                for size2 in range(min_modules, max_modules + 1):
                    if size1 == size2:
                        continue
                        
                    for size3 in range(min_modules, max_modules + 1):
                        if size3 == size1 or size3 == size2:
                            continue
                            
                        # Try various distributions with 3 string sizes
                        # Limit the search space to make computation feasible
                        max_count1 = min(20, (total_modules // size1) + 1)
                        for count1 in range(1, max_count1):
                            remaining1 = total_modules - (count1 * size1)
                            max_count2 = min(20, (remaining1 // size2) + 1)
                            
                            for count2 in range(1, max_count2):
                                remaining2 = remaining1 - (count2 * size2)
                                count3 = remaining2 // size3
                                unused = remaining2 % size3
                                
                                if count3 <= 0:
                                    continue
                                    
                                if unused < min_unused_modules:
                                    min_unused_modules = unused
                                    best_solution = [(count1, size1), (count2, size2), (count3, size3)]
                                    
                                    # If perfect solution, return immediately
                                    if unused == 0:
                                        return best_solution
    
    # If we still have unused modules and they're enough to make a separate string
    if min_unused_modules >= min_modules:
        if best_solution is not None:
            best_solution.append((1, min_unused_modules))
            min_unused_modules = 0
    
    # Handle case where we need one more string with fewer modules
    if min_unused_modules > 0 and min_unused_modules < min_modules:
        # Find if we can adjust existing strings to make one more valid string
        if best_solution:
            total_strings = sum(count for count, _ in best_solution)
            
            # Try to redistribute modules from existing strings
            # Approach 1: Redistribute evenly
            if total_strings >= min_modules - min_unused_modules:
                # We need to "steal" (min_modules - min_unused_modules) modules from existing strings
                modules_needed = min_modules - min_unused_modules
                
                # Make a copy of the solution to modify
                new_solution = []
                
                # Sort by module count descending so we reduce larger strings first
                sorted_solution = sorted(best_solution, key=lambda x: x[1], reverse=True)
                
                # Track how many modules we've redistributed
                redistributed = 0
                
                for count, size in sorted_solution:
                    # Calculate how many modules we can take from these strings
                    modules_per_string_to_remove = min(1, modules_needed // count)
                    if modules_per_string_to_remove > 0:
                        modules_redistributed = modules_per_string_to_remove * count
                        redistributed += modules_redistributed
                        modules_needed -= modules_redistributed
                        
                        # Add adjusted strings to new solution
                        new_solution.append((count, size - modules_per_string_to_remove))
                    else:
                        new_solution.append((count, size))
                
                if redistributed + min_unused_modules >= min_modules:
                    # We have enough modules to make a new string
                    new_solution.append((1, redistributed + min_unused_modules))
                    best_solution = new_solution
                    min_unused_modules = 0
    
    # Return the best solution we found
    return best_solution if best_solution else [(total_modules // min_modules, min_modules)]

def solar_mppt_allocator(azimuth_data, inverter_data, min_modules, max_modules, module_dc_power):
    """
    Calculate optimal allocation of strings to MPPTs across multiple azimuths
    
    Parameters:
    -----------
    azimuth_data : dict
        Dictionary with keys as azimuth names and values as total module count
        Example: {"North": 350, "South": 560, "East": 255, "West": 418}
    
    inverter_data : dict
        Dictionary with information about inverters
        Keys:
        - 'count': Number of inverters
        - 'mppt_per_inverter': Number of MPPTs per inverter
        - 'strings_per_mppt': Maximum strings per MPPT
        - 'ac_power': AC power rating of each inverter in kW
    
    min_modules : int
        Minimum modules allowed per string
    
    max_modules : int
        Maximum modules allowed per string
        
    module_dc_power : float
        DC power rating of each module in watts
    
    Returns:
    --------
    tuple : (allocation_df, summary_df, messages, string_config_df, azimuth_to_configurations, inverter_summary_df)
        - allocation_df: DataFrame showing MPPT allocation by azimuth
        - summary_df: Summary of allocation with utilization metrics
        - messages: List of warning or error messages
        - string_config_df: DataFrame showing string configurations by azimuth
        - azimuth_to_configurations: Dictionary mapping azimuths to configurations
        - inverter_summary_df: DataFrame showing utilization and DC/AC metrics by inverter
    """
    messages = []
    
    # Calculate optimal string configurations for each azimuth
    string_configurations = {}
    string_config_data = []
    
    for azimuth, total_modules in azimuth_data.items():
        configs = calculate_optimal_string_configuration(total_modules, min_modules, max_modules, inverter_data['strings_per_mppt'])
        
        # Store the configurations for allocation
        string_configurations[azimuth] = configs
        
        # Add to data for display
        for modules_per_string, string_count in [(c[1], c[0]) for c in configs]:
            string_config_data.append({
                'Azimuth': azimuth,
                'String Size': f"{modules_per_string} modules",
                'String Count': string_count,
                'Total Modules': string_count * modules_per_string
            })
    
    string_config_df = pd.DataFrame(string_config_data)
    
    # Calculate total strings needed across all configurations
    total_strings_needed = sum(count for config in string_configurations.values() 
                             for count, _ in config)
    
    # Calculate total capacity
    total_mppts = inverter_data['count'] * inverter_data['mppt_per_inverter']
    total_string_capacity = total_mppts * inverter_data['strings_per_mppt']
    
    if total_strings_needed > total_string_capacity:
        messages.append(f"Error: Total strings needed ({total_strings_needed}) exceeds capacity ({total_string_capacity})")
        return None, None, messages, string_config_df, None, None
    
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
    
    # Helper function to find available MPPTs
    def find_available_mppts(azimuth=None):
        available = []
        
        # Categorize inverters by utilization status
        unused_inverters = []  # Inverters with no MPPTs used
        partially_used_inverters = []  # Inverters with some MPPTs used
        
        for i, inv in enumerate(inverters):
            used_mppts = sum(1 for mppt in inv['mppts'] if mppt['azimuth'] is not None)
            if used_mppts == 0:
                unused_inverters.append((i, used_mppts))
            elif used_mppts < inverter_data['mppt_per_inverter']:
                partially_used_inverters.append((i, used_mppts))
        
        # Sort partially used inverters by usage (less used first)
        partially_used_inverters.sort(key=lambda x: x[1])
        
        # Prioritize finding MPPTs on partially used inverters first
        for source in [partially_used_inverters, unused_inverters]:
            for inv_idx, _ in source:
                inverter = inverters[inv_idx]
                for mppt_idx, mppt in enumerate(inverter['mppts']):
                    # If looking for a specific azimuth, only return MPPTs with that azimuth or empty ones
                    if azimuth is not None:
                        if mppt['azimuth'] == azimuth and mppt['strings'] < mppt['capacity']:
                            available.append((inv_idx, mppt_idx))
                    # Otherwise, return only empty MPPTs
                    elif mppt['azimuth'] is None:
                        available.append((inv_idx, mppt_idx))
            
            # If we found MPPTs in the current source category, we're done
            if available:
                break
        
        return available
    
    # Convert string configurations to a flat list for allocation
    flat_configurations = []
    for azimuth, configs in string_configurations.items():
        for string_count, modules_per_string in configs:
            flat_configurations.append((azimuth, string_count, modules_per_string))
    
    # Sort configurations by string count (descending)
    flat_configurations = sorted(flat_configurations, key=lambda x: x[1], reverse=True)
    
    # Group configurations by azimuth for better MPPT allocation
    azimuth_to_configurations = {}
    for azimuth, string_count, modules_per_string in flat_configurations:
        if azimuth not in azimuth_to_configurations:
            azimuth_to_configurations[azimuth] = []
        azimuth_to_configurations[azimuth].append((string_count, modules_per_string))
    
    # Sort azimuths by total string count (descending)
    sorted_azimuths = sorted(
        azimuth_to_configurations.items(), 
        key=lambda x: sum(count for count, _ in x[1]), 
        reverse=True
    )
    
    # Allocate strings to MPPTs - prioritize using all available MPPTs with fewer strings per MPPT rather than filling some MPPTs to capacity
    for azimuth, configurations in sorted_azimuths:
        # Calculate total strings for this azimuth
        total_azimuth_strings = sum(count for count, _ in configurations)
        
        # Calculate how many full MPPTs we need (with all strings_per_mppt strings)
        full_mppts_needed = total_azimuth_strings // inverter_data['strings_per_mppt']
        remaining_strings = total_azimuth_strings % inverter_data['strings_per_mppt']
        total_mppts_needed = full_mppts_needed + (1 if remaining_strings > 0 else 0)
        
        # Group configurations by module size for more efficient allocation
        size_to_strings = {}
        for string_count, module_size in configurations:
            if module_size not in size_to_strings:
                size_to_strings[module_size] = 0
            size_to_strings[module_size] += string_count
        
        # Sort module sizes by count (descending) for allocation
        sorted_sizes = sorted(size_to_strings.items(), key=lambda x: x[1], reverse=True)
        
        # Track DC power allocation per inverter for balanced loading
        inverter_dc_power = {}
        for inv_idx, inverter in enumerate(inverters):
            inv_modules = 0
            for mppt in inverter['mppts']:
                if mppt['azimuth'] is not None and mppt['strings'] > 0:
                    inv_modules += mppt['strings'] * mppt['modules_per_string']
            inverter_dc_power[inv_idx] = inv_modules * module_dc_power / 1000  # in kW
        
        # Allocate strings for each module size
        for module_size, string_count in sorted_sizes:
            strings_remaining = string_count
            
            while strings_remaining > 0:
                # Get a list of inverters with their current DC/AC ratios
                inverter_dc_ac_ratios = {}
                for inv_idx, inverter in enumerate(inverters):
                    # Calculate current DC power for this inverter
                    inv_modules = 0
                    for mppt in inverter['mppts']:
                        if mppt['azimuth'] is not None and mppt['strings'] > 0:
                            inv_modules += mppt['strings'] * mppt['modules_per_string']
                    
                    inv_dc_power = inv_modules * module_dc_power / 1000  # in kW
                    inv_dc_ac_ratio = (inv_dc_power / inverter_data['ac_power']) * 100
                    inverter_dc_ac_ratios[inv_idx] = inv_dc_ac_ratio
                
                # Sort inverters by DC/AC ratio (ascending) to prioritize less loaded inverters
                sorted_inverters = sorted(inverter_dc_ac_ratios.items(), key=lambda x: x[1])
                
                # Find an appropriate MPPT in the least loaded inverter that has space
                found_mppt = False
                
                for inv_idx, current_dc_ac_ratio in sorted_inverters:
                    # Skip inverters that are already at or near the 120% limit
                    if current_dc_ac_ratio >= (inverter_data['max_inverter_dc_ac'] - 2):  # -2% buffer
                        continue
                    
                    inverter = inverters[inv_idx]
                    max_inverter_dc_ac = inverter_data['max_inverter_dc_ac']  # Maximum DC/AC ratio for the inverter
                    
                    # Calculate DC power of one string with this module size
                    string_dc_power = module_size * module_dc_power / 1000  # in kW
                    
                    # Find empty MPPTs in this inverter
                    available_mppts_in_inverter = []
                    for mppt_idx, mppt in enumerate(inverter['mppts']):
                        if mppt['azimuth'] is None:
                            available_mppts_in_inverter.append(mppt_idx)
                        elif mppt['azimuth'] == azimuth and mppt['strings'] < mppt['capacity']:
                            available_mppts_in_inverter.append(mppt_idx)
                    
                    if not available_mppts_in_inverter:
                        continue
                    
                    # Calculate how many strings we can add to this inverter before hitting DC/AC limit
                    inv_ac_power = inverter_data['ac_power']
                    current_inv_dc_power = current_dc_ac_ratio * inv_ac_power / 100
                    max_inv_dc_power = max_inverter_dc_ac * inv_ac_power / 100
                    available_dc_power = max_inv_dc_power - current_inv_dc_power
                    max_strings_for_inverter = int(available_dc_power / string_dc_power)
                    
                    # Limit by remaining strings
                    strings_to_add_to_inverter = min(max_strings_for_inverter, strings_remaining)
                    
                    if strings_to_add_to_inverter <= 0:
                        continue
                    
                    # Now distribute these strings across available MPPTs in this inverter
                    # First, prioritize MPPTs that already have this azimuth
                    for priority in ['existing', 'empty']:
                        if strings_to_add_to_inverter <= 0:
                            break
                        
                        for mppt_idx in available_mppts_in_inverter:
                            mppt = inverter['mppts'][mppt_idx]
                            
                            # Skip if we're looking for existing azimuth MPPTs but this is empty
                            if priority == 'existing' and mppt['azimuth'] != azimuth:
                                continue
                                
                            # Skip if we're looking for empty MPPTs but this has an azimuth already
                            if priority == 'empty' and mppt['azimuth'] is not None:
                                continue
                            
                            # Calculate MPPT capacity and current load
                            mppt_strings_capacity = mppt['capacity']  # Usually 3 strings
                            current_mppt_strings = mppt['strings']
                            
                            # Calculate DC power for current strings in this MPPT
                            current_mppt_dc_power = 0
                            if current_mppt_strings > 0:
                                current_mppt_dc_power = current_mppt_strings * mppt['modules_per_string'] * module_dc_power / 1000
                            
                            # Calculate DC/AC ratio for this MPPT (based on its portion of inverter AC)
                            mppt_ac_power = inverter_data['ac_power'] / inverter_data['mppt_per_inverter']
                            current_mppt_dc_ac = (current_mppt_dc_power / mppt_ac_power) * 100
                            
                            # Allow up to configured maximum loading for each MPPT
                            max_mppt_dc_ac = inverter_data['max_mppt_dc_ac']
                            max_mppt_dc_power = mppt_ac_power * max_mppt_dc_ac / 100
                            available_mppt_dc_power = max_mppt_dc_power - current_mppt_dc_power
                            
                            # Calculate how many strings we can add to this MPPT
                            max_strings_for_mppt = min(
                                int(available_mppt_dc_power / string_dc_power),  # Limited by DC/AC
                                mppt_strings_capacity - current_mppt_strings,    # Limited by physical capacity
                                strings_to_add_to_inverter                       # Limited by inverter DC/AC
                            )
                            
                            if max_strings_for_mppt <= 0:
                                continue
                                
                            # Add strings to this MPPT
                            if mppt['azimuth'] is None:
                                mppt['azimuth'] = azimuth
                                mppt['modules_per_string'] = module_size
                                mppt['strings'] = max_strings_for_mppt
                            else:
                                # Only add strings if module size matches
                                if mppt['modules_per_string'] == module_size:
                                    mppt['strings'] += max_strings_for_mppt
                            
                            strings_remaining -= max_strings_for_mppt
                            strings_to_add_to_inverter -= max_strings_for_mppt
                            found_mppt = True
                            
                            if strings_remaining <= 0 or strings_to_add_to_inverter <= 0:
                                break
                
                if not found_mppt:
                    messages.append(f"Error: Could not find suitable MPPT for {azimuth} with {module_size} modules per string. {strings_remaining} strings remaining.")
                    break
        
                if strings_remaining <= 0:
                    break
    
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
    for azimuth, total_modules in azimuth_data.items():
        azimuth_rows = allocation_df[allocation_df['Assigned Azimuth'] == azimuth]
        mppts_used = len(azimuth_rows)
        allocated_strings = azimuth_rows['Strings'].sum()
        allocated_modules = azimuth_rows['Total Modules'].sum()
        
        # Calculate DC power and DC/AC ratio
        dc_power = allocated_modules * module_dc_power / 1000  # in kW
        dc_ac_ratio = (dc_power / (inverter_data['count'] * inverter_data['ac_power'])) * 100
        
        # Calculate waste percentage
        waste_percentage = 0
        if total_modules > 0:
            waste_percentage = ((total_modules - allocated_modules) / total_modules) * 100
        
        # Calculate MPPT utilization
        mppt_capacity = mppts_used * inverter_data['strings_per_mppt']
        mppt_utilization = (allocated_strings / mppt_capacity) * 100 if mppt_capacity > 0 else 0
        
        # Calculate total available MPPTs and the percentage used
        total_mppts = inverter_data['count'] * inverter_data['mppt_per_inverter']
        mppt_usage_pct = (mppts_used / total_mppts) * 100
        
        # Check if all modules were allocated
        if allocated_modules != total_modules:
            messages.append(f"Warning: {azimuth} was supposed to have {total_modules} modules but {allocated_modules} were allocated.")
        
        summary_data.append({
            'Azimuth': azimuth,
            'Total Strings': allocated_strings,
            'Total Modules': allocated_modules,
            'Requested Modules': total_modules,
            'DC Power (kW)': round(dc_power, 2),
            'DC/AC Ratio (%)': round(dc_ac_ratio, 2),
            'Waste (%)': round(waste_percentage, 2),
            'MPPTs Used': mppts_used,
            'Total MPPTs': total_mppts,
            'MPPT Usage (%)': round(mppt_usage_pct, 2),
            'MPPT Utilization (%)': round(mppt_utilization, 1)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add system total row
    total_requested = sum(total_modules for total_modules in azimuth_data.values())
    total_allocated = summary_df['Total Modules'].sum()
    total_waste = ((total_requested - total_allocated) / total_requested) * 100 if total_requested > 0 else 0
    total_mppts_used = len(allocation_df[allocation_df['Assigned Azimuth'] != 'Unused'])
    total_mppts = inverter_data['count'] * inverter_data['mppt_per_inverter']
    total_mppt_usage = (total_mppts_used / total_mppts) * 100
    total_mppt_capacity = total_mppts_used * inverter_data['strings_per_mppt']
    total_mppt_utilization = (summary_df['Total Strings'].sum() / total_mppt_capacity) * 100 if total_mppt_capacity > 0 else 0
    total_dc_power = total_allocated * module_dc_power / 1000  # in kW
    total_ac_power = inverter_data['count'] * inverter_data['ac_power']
    total_dc_ac_ratio = (total_dc_power / total_ac_power) * 100 if total_ac_power > 0 else 0
    
    system_total = {
        'Azimuth': 'SYSTEM TOTAL',
        'Total Strings': summary_df['Total Strings'].sum(),
        'Total Modules': total_allocated,
        'Requested Modules': total_requested,
        'DC Power (kW)': round(total_dc_power, 2),
        'DC/AC Ratio (%)': round(total_dc_ac_ratio, 2),
        'Waste (%)': round(total_waste, 2),
        'MPPTs Used': total_mppts_used,
        'Total MPPTs': total_mppts,
        'MPPT Usage (%)': round(total_mppt_usage, 2),
        'MPPT Utilization (%)': round(total_mppt_utilization, 1)
    }
    
    # Convert all columns to string type for consistent display and Arrow compatibility
    summary_df = summary_df.astype({
        'Total Strings': 'int64',
        'Total Modules': 'int64',
        'MPPTs Used': 'int64',
        'MPPT Usage (%)': 'str',
        'MPPT Utilization (%)': 'str'
    })
    
    # Format the 'MPPT Usage (%)' column to string type with 2 decimal places
    summary_df['MPPT Usage (%)'] = summary_df['MPPT Usage (%)'].apply(
        lambda x: f"{float(x):.2f}" if isinstance(x, (int, float)) else x
    )
    
    # Now add the system total row
    system_total_df = pd.DataFrame([system_total])
    system_total_df = system_total_df.astype({
        'Total Strings': 'int64',
        'Total Modules': 'int64',
        'MPPTs Used': 'int64',
        'MPPT Usage (%)': 'str',
        'MPPT Utilization (%)': 'str'
    })
    
    summary_df = pd.concat([summary_df, system_total_df], ignore_index=True)
    
    # Convert any None values to appropriate string representation after concatenation
    summary_df = summary_df.fillna('-')
    
    # Calculate inverter summary
    inverter_summary_data = []
    for inv_idx, inverter in enumerate(inverters):
        inverter_id = inverter['id']
        # Calculate modules and DC power for this inverter
        inverter_modules = 0
        inverter_mppts_used = 0
        for mppt in inverter['mppts']:
            if mppt['azimuth'] is not None and mppt['azimuth'] != 'Unused':
                inverter_modules += mppt['strings'] * mppt['modules_per_string']
                if mppt['strings'] > 0:
                    inverter_mppts_used += 1
        
        # Calculate DC power and DC/AC ratio
        inverter_dc_power = inverter_modules * module_dc_power / 1000  # Convert to kW
        dc_ac_ratio = (inverter_dc_power / inverter_data['ac_power']) * 100 if inverter_data['ac_power'] > 0 else 0
        
        # Calculate MPPT utilization
        total_mppts = inverter_data['mppt_per_inverter']
        mppt_utilization_pct = (inverter_mppts_used / total_mppts) * 100
        
        inverter_summary_data.append({
            'Inverter': inverter_id,
            'Modules': inverter_modules,
            'DC Power (kW)': round(inverter_dc_power, 2),
            'AC Power (kW)': inverter_data['ac_power'],
            'DC/AC Ratio (%)': round(dc_ac_ratio, 2),
            'MPPTs Used': inverter_mppts_used,
            'Total MPPTs': total_mppts,
            'MPPT Utilization (%)': round(mppt_utilization_pct, 1)
        })
    
    inverter_summary_df = pd.DataFrame(inverter_summary_data)
    
    return allocation_df, summary_df, messages, string_config_df, azimuth_to_configurations, inverter_summary_df

# Streamlit app
st.set_page_config(page_title="Solar PV String-to-MPPT Calculator", page_icon="☀️", layout="wide")

st.title("☀️ Solar PV String-to-MPPT Allocation Tool")
st.write("""
This tool helps solar PV design engineers allocate strings to MPPTs across multiple azimuths.
It calculates optimal string configurations based on total module count and respects constraints on modules per string.
Each MPPT will only contain strings from a single azimuth.
""")

# Create sidebar for inputs
st.sidebar.header("System Configuration")

# Inverter configuration
st.sidebar.subheader("Inverter Configuration")
inverter_count = st.sidebar.number_input("Number of Inverters", min_value=1, max_value=50, value=9)
mppt_per_inverter = st.sidebar.number_input("MPPTs per Inverter", min_value=1, max_value=20, value=7)
strings_per_mppt = st.sidebar.number_input("Strings per MPPT", min_value=1, max_value=10, value=3)
inverter_ac_power = st.sidebar.number_input("Inverter AC Power Rating (kW)", min_value=1.0, max_value=1000.0, value=150.0)
total_ac_power = inverter_count * inverter_ac_power

# Module constraints
st.sidebar.subheader("Module Constraints")
min_modules = st.sidebar.number_input("Minimum Modules per String", min_value=1, max_value=30, value=15)
max_modules = st.sidebar.number_input("Maximum Modules per String", min_value=min_modules, max_value=30, value=19)
module_dc_power = st.sidebar.number_input("Module DC Power Rating (W)", min_value=100, max_value=1000, value=540)

# DC/AC Ratio Target
st.sidebar.subheader("DC/AC Ratio Target")
inverter_dc_ac_min, inverter_dc_ac_max = st.sidebar.slider(
    "Target Inverter DC/AC Ratio Range (%)", 
    min_value=80, 
    max_value=140, 
    value=(90, 120),
    help="Maximum DC/AC ratio allowed for each inverter (120% ±2% enforced)"
)

mppt_dc_ac_max = st.sidebar.slider(
    "Maximum MPPT DC/AC Ratio (%)",
    min_value=100,
    max_value=180,
    value=140,
    help="Maximum DC/AC ratio allowed for individual MPPTs (can be higher than inverter limit)"
)

# Use middle of selected range as target for overall system
target_dc_ac_ratio = (inverter_dc_ac_min + inverter_dc_ac_max) / 2

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
        total_modules = st.number_input(f"Total Modules", min_value=1, value=360, key=f"modules_{i}")
        
        azimuth_data[azimuth_name] = total_modules

# Create a calculate button
if st.button("Calculate Allocation"):
    # Prepare data for calculation
    inverter_data = {
        'count': inverter_count,
        'mppt_per_inverter': mppt_per_inverter,
        'strings_per_mppt': strings_per_mppt,
        'ac_power': inverter_ac_power,
        'target_dc_ac_ratio': target_dc_ac_ratio,
        'max_inverter_dc_ac': 120,  # Hard limit at 120%
        'max_mppt_dc_ac': mppt_dc_ac_max  # From slider
    }
    
    # Run the calculator
    allocation_df, summary_df, messages, string_config_df, azimuth_to_configurations, inverter_summary_df = solar_mppt_allocator(azimuth_data, inverter_data, min_modules, max_modules, module_dc_power)
    
    # Display any warnings or errors
    if messages:
        st.warning("Warnings/Errors:")
        for msg in messages:
            st.write(f"- {msg}")
    
    if allocation_df is not None and summary_df is not None:
        # Display string configurations
        st.header("String Configurations")
        
        # Enhance string config display
        enhanced_string_config = []
        for azimuth, configs in azimuth_to_configurations.items():
            azimuth_total_strings = sum(count for count, _ in configs)
            azimuth_total_modules = sum(count * size for count, size in configs)
            
            # Format configuration as a readable string
            config_str = " + ".join([f"{count} x {size} modules" for count, size in configs])
            
            enhanced_string_config.append({
                'Azimuth': azimuth,
                'Configuration': config_str,
                'Total Strings': azimuth_total_strings,
                'Total Modules': azimuth_total_modules,
                'Requested Modules': azimuth_data[azimuth],
                'Waste (%)': round(((azimuth_data[azimuth] - azimuth_total_modules) / azimuth_data[azimuth]) * 100, 2) if azimuth_data[azimuth] > 0 else 0,
                'MPPTs Required': math.ceil(azimuth_total_strings / inverter_data['strings_per_mppt'])
            })
        
        enhanced_config_df = pd.DataFrame(enhanced_string_config)
        st.dataframe(enhanced_config_df)
        
        # Display a more detailed breakdown if needed
        with st.expander("Detailed String Configuration Breakdown"):
            st.dataframe(string_config_df)
        
        # Display summary
        st.header("Summary")
        st.dataframe(summary_df)
        
        # Display per-inverter summary
        st.header("Inverter Summary")
        st.dataframe(inverter_summary_df)
        
        # Display allocation
        st.header("MPPT Allocation")
        st.dataframe(allocation_df)
        
        # Prepare Excel with all sheets for download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            enhanced_config_df.to_excel(writer, sheet_name='String Configurations', index=False)
            string_config_df.to_excel(writer, sheet_name='Detailed Configurations', index=False)
            allocation_df.to_excel(writer, sheet_name='MPPT Allocation', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            inverter_summary_df.to_excel(writer, sheet_name='Inverter Summary', index=False)
        
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
            st.write(f"**{azimuth}** - {len(azimuth_allocation)} MPPTs, {azimuth_allocation['Strings'].sum()} strings, {azimuth_allocation['Total Modules'].sum()} modules")
            st.dataframe(azimuth_allocation[['Inverter', 'MPPT', 'Strings', 'Modules per String', 'Total Modules']])
