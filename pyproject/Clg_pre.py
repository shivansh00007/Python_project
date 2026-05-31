# ==============================================================================
# NORTH INDIA COLLEGE ADMISSION PREDICTOR - Enhanced Edition v2.0
# Home.py — Main predictor with all 4 streams
# ==============================================================================

import streamlit as st
import numpy as np
import joblib
import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

NORTH_STATES = ['Delhi', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Uttarakhand',
                 'Rajasthan', 'Himachal Pradesh', 'Jammu & Kashmir']
CATEGORIES   = ['General', 'OBC-NCL', 'SC', 'ST', 'EWS']
CATEGORY_MAP = {'General': 0, 'OBC-NCL': 1, 'SC': 2, 'ST': 3, 'EWS': 4}

# ==============================================================================
# COLLEGE DETAIL DATABASE
# ==============================================================================

COLLEGE_DETAILS = {
    "IIT Delhi": {
        "established": 1961, "type": "Central / IIT", "city": "New Delhi",
        "total_students": 8500, "faculty": 600, "nirf_rank": 2, "naac": "A++",
        "avg_package_lpa": 28.5, "highest_package_lpa": 230,
        "placement_pct": 95, "top_recruiters": "Google, Microsoft, Goldman Sachs, McKinsey",
        "fee_per_year_lpa": 2.25, "campus_acres": 325,
        "known_for": "Research excellence, Silicon Valley placements, elite alumni network",
        "roi_score": 98, "roi_rationale": "Avg salary recoups 4-yr fees in under 3 months",
        "hostel": "Available (mandatory 1st year)", "sports": "Olympic-level facilities",
        "research_centers": 25, "international_collab": "MIT, Stanford, ETH Zurich"
    },
    "IIT Kanpur": {
        "established": 1959, "type": "Central / IIT", "city": "Kanpur, UP",
        "total_students": 7200, "faculty": 480, "nirf_rank": 5, "naac": "A++",
        "avg_package_lpa": 24, "highest_package_lpa": 180,
        "placement_pct": 93, "top_recruiters": "Amazon, Apple, ISRO, Boston Consulting",
        "fee_per_year_lpa": 2.2, "campus_acres": 1055,
        "known_for": "CS + Physics excellence, aerospace & space research",
        "roi_score": 97, "roi_rationale": "Massive 1055-acre campus, world-class research, very high salaries",
        "hostel": "Available (all years)", "sports": "Cricket, Football, Swimming pool",
        "research_centers": 20, "international_collab": "Caltech, TU Munich"
    },
    "IIT Roorkee": {
        "established": 1847, "type": "Central / IIT", "city": "Roorkee, Uttarakhand",
        "total_students": 9000, "faculty": 550, "nirf_rank": 7, "naac": "A++",
        "avg_package_lpa": 22, "highest_package_lpa": 150,
        "placement_pct": 92, "top_recruiters": "Intel, Adobe, DE Shaw, L&T, ONGC",
        "fee_per_year_lpa": 2.2, "campus_acres": 365,
        "known_for": "Oldest tech institute in Asia, civil & structural engineering legacy",
        "roi_score": 96, "roi_rationale": "Heritage brand + strong placements = outstanding lifetime ROI",
        "hostel": "Available", "sports": "India's best college sports infrastructure",
        "research_centers": 18, "international_collab": "IIT consortium, DAAD Germany"
    },
    "IIT BHU Varanasi": {
        "established": 1919, "type": "Central / IIT", "city": "Varanasi, UP",
        "total_students": 6500, "faculty": 360, "nirf_rank": 10, "naac": "A+",
        "avg_package_lpa": 18, "highest_package_lpa": 120,
        "placement_pct": 88, "top_recruiters": "TCS, Wipro, Samsung, Deloitte, BHEL",
        "fee_per_year_lpa": 2.1, "campus_acres": 1300,
        "known_for": "Inside BHU campus, metallurgy & mining strength, cultural richness",
        "roi_score": 92, "roi_rationale": "Good salary + very low fees + historic brand = solid ROI",
        "hostel": "Available (1300-acre BHU campus)", "sports": "Full BHU sports complex",
        "research_centers": 15, "international_collab": "UNESCO partnerships"
    },
    "IIT Ropar": {
        "established": 2008, "type": "Central / IIT", "city": "Rupnagar, Punjab",
        "total_students": 2800, "faculty": 150, "nirf_rank": 28, "naac": "A",
        "avg_package_lpa": 16, "highest_package_lpa": 95,
        "placement_pct": 85, "top_recruiters": "Qualcomm, IBM, Infosys, EXL Analytics",
        "fee_per_year_lpa": 2.1, "campus_acres": 500,
        "known_for": "New-gen IIT, growing research ecosystem, Punjab industrial belt access",
        "roi_score": 88, "roi_rationale": "Emerging IIT — ROI improving year-on-year rapidly",
        "hostel": "Available", "sports": "Modern facilities",
        "research_centers": 8, "international_collab": "Growing international MoUs"
    },
    "IIT Jodhpur": {
        "established": 2008, "type": "Central / IIT", "city": "Jodhpur, Rajasthan",
        "total_students": 2500, "faculty": 140, "nirf_rank": 31, "naac": "A",
        "avg_package_lpa": 15, "highest_package_lpa": 90,
        "placement_pct": 84, "top_recruiters": "DRDO, L&T, Tata Motors, Accenture",
        "fee_per_year_lpa": 2.1, "campus_acres": 852,
        "known_for": "Desert campus, renewable energy & solar research hub",
        "roi_score": 86, "roi_rationale": "Good niche ROI especially for core engineering and defence roles",
        "hostel": "Available", "sports": "Desert campus facilities",
        "research_centers": 7, "international_collab": "MNRE partnerships"
    },
    "IIT Mandi": {
        "established": 2009, "type": "Central / IIT", "city": "Mandi, Himachal Pradesh",
        "total_students": 2200, "faculty": 120, "nirf_rank": 40, "naac": "A",
        "avg_package_lpa": 14, "highest_package_lpa": 80,
        "placement_pct": 82, "top_recruiters": "HPCL, Infosys, Samsung, Honeywell",
        "fee_per_year_lpa": 2.0, "campus_acres": 529,
        "known_for": "Himalayan campus, smart cities & disaster tech research",
        "roi_score": 84, "roi_rationale": "Low fees + decent packages = good ROI for core engineering",
        "hostel": "Available (scenic Himalayan location)", "sports": "Trekking, outdoor sports",
        "research_centers": 6, "international_collab": "HP state govt research"
    },
    "IIT Jammu": {
        "established": 2016, "type": "Central / IIT", "city": "Jammu, J&K",
        "total_students": 1500, "faculty": 90, "nirf_rank": 55, "naac": "A",
        "avg_package_lpa": 12, "highest_package_lpa": 65,
        "placement_pct": 78, "top_recruiters": "NIC, BSNL, Wipro, TCS, JKPSC",
        "fee_per_year_lpa": 1.9, "campus_acres": 350,
        "known_for": "Newest IIT, gateway for J&K engineering talent",
        "roi_score": 80, "roi_rationale": "Early-stage IIT — ROI building, strong government job links",
        "hostel": "Available", "sports": "Basic facilities, expanding",
        "research_centers": 4, "international_collab": "Central Univ J&K partnerships"
    },
    "MNNIT Allahabad": {
        "established": 1961, "type": "Central / NIT", "city": "Allahabad, UP",
        "total_students": 5500, "faculty": 280, "nirf_rank": 50, "naac": "A+",
        "avg_package_lpa": 12, "highest_package_lpa": 55,
        "placement_pct": 85, "top_recruiters": "TCS, HCL, NTPC, Infosys, BHEL",
        "fee_per_year_lpa": 1.4, "campus_acres": 220,
        "known_for": "Heritage NIT on Ganga banks, strong alumni network across UP/Delhi",
        "roi_score": 89, "roi_rationale": "Very low fees vs packages = excellent ROI, better than many private colleges",
        "hostel": "Available (boys & girls)", "sports": "Full facilities on Sangam campus",
        "research_centers": 10, "international_collab": "NIT consortium"
    },
    "MNIT Jaipur": {
        "established": 1963, "type": "Central / NIT", "city": "Jaipur, Rajasthan",
        "total_students": 5200, "faculty": 260, "nirf_rank": 52, "naac": "A+",
        "avg_package_lpa": 11.5, "highest_package_lpa": 48,
        "placement_pct": 84, "top_recruiters": "Amazon, Flipkart, TATA, Deloitte",
        "fee_per_year_lpa": 1.35, "campus_acres": 317,
        "known_for": "Pink City location, strong CS + Electronics, growing startup culture",
        "roi_score": 88, "roi_rationale": "Jaipur tech scene growing fast — solid and improving placement ROI",
        "hostel": "Available", "sports": "Cricket, Basketball, Athletic track",
        "research_centers": 9, "international_collab": "CSIR Jaipur"
    },
    "NIT Kurukshetra": {
        "established": 1963, "type": "Central / NIT", "city": "Kurukshetra, Haryana",
        "total_students": 5800, "faculty": 270, "nirf_rank": 48, "naac": "A+",
        "avg_package_lpa": 12.5, "highest_package_lpa": 60,
        "placement_pct": 87, "top_recruiters": "Samsung, Qualcomm, Microsoft, L&T, Maruti",
        "fee_per_year_lpa": 1.4, "campus_acres": 300,
        "known_for": "Close to Delhi NCR, strong IT + Electronics, Haryana industry links",
        "roi_score": 90, "roi_rationale": "NCR proximity + good packages = top-tier ROI among all NITs",
        "hostel": "Available (boys & girls)", "sports": "Full sports complex",
        "research_centers": 11, "international_collab": "IET UK chapter"
    },
    "IIIT Allahabad": {
        "established": 1999, "type": "Central / IIIT", "city": "Allahabad, UP",
        "total_students": 3500, "faculty": 180, "nirf_rank": 60, "naac": "A",
        "avg_package_lpa": 14, "highest_package_lpa": 70,
        "placement_pct": 90, "top_recruiters": "Google, Microsoft, Adobe, Goldman Sachs",
        "fee_per_year_lpa": 1.5, "campus_acres": 100,
        "known_for": "Pure IT/CS focus, one of best IIITs, consistently top CS placements",
        "roi_score": 93, "roi_rationale": "CS-only = very high placement rate + premium packages = excellent ROI",
        "hostel": "Available", "sports": "Basic facilities",
        "research_centers": 8, "international_collab": "IEEE, ACM chapters"
    },
    "NIT Jalandhar": {
        "established": 1987, "type": "Central / NIT", "city": "Jalandhar, Punjab",
        "total_students": 5000, "faculty": 230, "nirf_rank": 65, "naac": "A",
        "avg_package_lpa": 10, "highest_package_lpa": 45,
        "placement_pct": 82, "top_recruiters": "TCS, Infosys, HCL, Wipro, Hero MotoCorp",
        "fee_per_year_lpa": 1.3, "campus_acres": 250,
        "known_for": "Punjab industrial belt, good core engineering placements",
        "roi_score": 85, "roi_rationale": "Low fees + Punjab industry access = decent ROI",
        "hostel": "Available", "sports": "Full facilities",
        "research_centers": 7, "international_collab": "NIT network"
    },
    "NIT Delhi": {
        "established": 2010, "type": "Central / NIT", "city": "New Delhi",
        "total_students": 2200, "faculty": 110, "nirf_rank": 70, "naac": "A",
        "avg_package_lpa": 11, "highest_package_lpa": 50,
        "placement_pct": 83, "top_recruiters": "Deloitte, IBM, Capgemini, BSNL",
        "fee_per_year_lpa": 1.5, "campus_acres": 70,
        "known_for": "Delhi location = massive internship + placement advantage",
        "roi_score": 87, "roi_rationale": "Delhi location alone adds massive career ROI through networking",
        "hostel": "Limited (most students live off-campus)", "sports": "Basic",
        "research_centers": 5, "international_collab": "Delhi tech ecosystem"
    },
    "NIT Hamirpur": {
        "established": 1986, "type": "Central / NIT", "city": "Hamirpur, HP",
        "total_students": 4200, "faculty": 200, "nirf_rank": 75, "naac": "A",
        "avg_package_lpa": 9, "highest_package_lpa": 40,
        "placement_pct": 80, "top_recruiters": "TCS, Wipro, HPPCL, BSNL, NBCC",
        "fee_per_year_lpa": 1.3, "campus_acres": 280,
        "known_for": "Himalayan location, civil + electrical strength, peaceful campus",
        "roi_score": 82, "roi_rationale": "Low cost of living + low fees = decent overall ROI",
        "hostel": "Available (all years)", "sports": "Good outdoor sports in hills",
        "research_centers": 6, "international_collab": "HP state partnerships"
    },
    "NIT Uttarakhand": {
        "established": 2009, "type": "Central / NIT", "city": "Srinagar, Uttarakhand",
        "total_students": 1800, "faculty": 90, "nirf_rank": 90, "naac": "B+",
        "avg_package_lpa": 8, "highest_package_lpa": 35,
        "placement_pct": 75, "top_recruiters": "ONGC, TCS, Wipro, UJVNL",
        "fee_per_year_lpa": 1.25, "campus_acres": 200,
        "known_for": "Pahari campus on Alaknanda river, growing infrastructure",
        "roi_score": 78, "roi_rationale": "Newer NIT still building — ROI moderate but improving",
        "hostel": "Available", "sports": "River activities, trekking",
        "research_centers": 3, "international_collab": "Developing"
    },
    "NIT Srinagar": {
        "established": 1960, "type": "Central / NIT", "city": "Srinagar, J&K",
        "total_students": 3500, "faculty": 175, "nirf_rank": 80, "naac": "A",
        "avg_package_lpa": 9, "highest_package_lpa": 38,
        "placement_pct": 78, "top_recruiters": "JKPDD, BSNL, TCS, HCL, NHPC",
        "fee_per_year_lpa": 1.3, "campus_acres": 650,
        "known_for": "Kashmir valley campus, hydro-power & civil engineering",
        "roi_score": 81, "roi_rationale": "Strong J&K government job access = good regional ROI",
        "hostel": "Available (scenic Dal Lake region)", "sports": "Full facilities",
        "research_centers": 6, "international_collab": "J&K govt projects"
    },
    "DTU Delhi": {
        "established": 1941, "type": "State / Delhi Govt", "city": "New Delhi",
        "total_students": 9000, "faculty": 450, "nirf_rank": 36, "naac": "A+",
        "avg_package_lpa": 15, "highest_package_lpa": 90,
        "placement_pct": 88, "top_recruiters": "Microsoft, Amazon, Accenture, KPMG, Deloitte",
        "fee_per_year_lpa": 1.8, "campus_acres": 164,
        "known_for": "Delhi's #1 state engineering college, huge alumni network in Delhi NCR",
        "roi_score": 94, "roi_rationale": "Delhi location + top packages + moderate fees = one of best ROIs in North India",
        "hostel": "Available (boys & girls)", "sports": "Full sports complex",
        "research_centers": 14, "international_collab": "TU Berlin, NUS Singapore"
    },
    "NSUT Delhi": {
        "established": 1983, "type": "State / Delhi Govt", "city": "New Delhi",
        "total_students": 7500, "faculty": 380, "nirf_rank": 45, "naac": "A+",
        "avg_package_lpa": 13, "highest_package_lpa": 75,
        "placement_pct": 86, "top_recruiters": "Samsung, Cognizant, Infosys, L&T, Schlumberger",
        "fee_per_year_lpa": 1.75, "campus_acres": 145,
        "known_for": "Strong Electronics + CS, Delhi location, sister college of DTU",
        "roi_score": 92, "roi_rationale": "Delhi NCR location multiplies career value significantly",
        "hostel": "Available", "sports": "Good facilities",
        "research_centers": 10, "international_collab": "Delhi Skill Mission"
    },
    "IGDTUW Delhi (Women)": {
        "established": 1998, "type": "State / Delhi Govt (Women)", "city": "New Delhi",
        "total_students": 3500, "faculty": 200, "nirf_rank": 55, "naac": "A",
        "avg_package_lpa": 11, "highest_package_lpa": 55,
        "placement_pct": 84, "top_recruiters": "Accenture, TCS, Deloitte, Capgemini",
        "fee_per_year_lpa": 1.6, "campus_acres": 60,
        "known_for": "India's largest women's engineering college, empowering women in tech",
        "roi_score": 88, "roi_rationale": "Excellent for women — Delhi location + focused placement support",
        "hostel": "Available (women only campus)", "sports": "Full women's sports facilities",
        "research_centers": 7, "international_collab": "UN Women partnerships"
    },
    "PEC Chandigarh": {
        "established": 1921, "type": "State / Punjab Govt", "city": "Chandigarh",
        "total_students": 4500, "faculty": 220, "nirf_rank": 62, "naac": "A",
        "avg_package_lpa": 10.5, "highest_package_lpa": 45,
        "placement_pct": 82, "top_recruiters": "Maruti, BHEL, BSNL, TCS, Hero Honda",
        "fee_per_year_lpa": 1.5, "campus_acres": 75,
        "known_for": "Heritage Chandigarh campus, strong Punjab alumni network",
        "roi_score": 85, "roi_rationale": "Chandigarh location + Punjab industry = solid ROI",
        "hostel": "Available", "sports": "Full Chandigarh city access",
        "research_centers": 8, "international_collab": "Punjab Govt tech programs"
    },
    "HBTU Kanpur": {
        "established": 1966, "type": "State / UP Govt", "city": "Kanpur, UP",
        "total_students": 5000, "faculty": 240, "nirf_rank": 78, "naac": "A",
        "avg_package_lpa": 9, "highest_package_lpa": 38,
        "placement_pct": 78, "top_recruiters": "BPCL, HPCL, Reliance, TCS, Wipro",
        "fee_per_year_lpa": 1.2, "campus_acres": 200,
        "known_for": "Kanpur industrial belt, leather & chemical engineering strength",
        "roi_score": 83, "roi_rationale": "Very low fees + Kanpur industry access = decent ROI",
        "hostel": "Available", "sports": "Good facilities",
        "research_centers": 6, "international_collab": "IIT Kanpur proximity"
    },
    "Thapar University": {
        "established": 1956, "type": "Private / Deemed", "city": "Patiala, Punjab",
        "total_students": 12000, "faculty": 600, "nirf_rank": 27, "naac": "A+",
        "avg_package_lpa": 14, "highest_package_lpa": 80,
        "placement_pct": 91, "top_recruiters": "Google, Microsoft, Nvidia, Goldman Sachs, Samsung",
        "fee_per_year_lpa": 3.5, "campus_acres": 250,
        "known_for": "Best private engineering college in North India, Silicon Valley network",
        "roi_score": 85, "roi_rationale": "Higher fees but top packages — strong ROI especially for CS",
        "hostel": "Available (world-class)", "sports": "Olympic-standard facilities",
        "research_centers": 20, "international_collab": "UBC Canada, TU Delft"
    },
    "JIIT Noida": {
        "established": 2001, "type": "Private", "city": "Noida, UP",
        "total_students": 8500, "faculty": 420, "nirf_rank": 95, "naac": "A",
        "avg_package_lpa": 9, "highest_package_lpa": 42,
        "placement_pct": 82, "top_recruiters": "Infosys, TCS, Wipro, HCL, Cognizant",
        "fee_per_year_lpa": 2.8, "campus_acres": 100,
        "known_for": "Noida tech hub location, Jaypee group, strong IT focus",
        "roi_score": 72, "roi_rationale": "Noida location helps but high fees vs packages reduces ROI",
        "hostel": "Available", "sports": "Good facilities",
        "research_centers": 8, "international_collab": "Noida SEZ companies"
    },
    "Shiv Nadar University": {
        "established": 2011, "type": "Private / Deemed", "city": "Greater Noida, UP",
        "total_students": 5000, "faculty": 380, "nirf_rank": 62, "naac": "A+",
        "avg_package_lpa": 16, "highest_package_lpa": 95,
        "placement_pct": 89, "top_recruiters": "Google, Amazon, HCL, JP Morgan, KPMG",
        "fee_per_year_lpa": 4.2, "campus_acres": 286,
        "known_for": "HCL founder's university, research-focused, international faculty",
        "roi_score": 80, "roi_rationale": "High fees balanced by premium packages — good ROI for top performers",
        "hostel": "Available (premium)", "sports": "World-class",
        "research_centers": 15, "international_collab": "Yale, NUS, Imperial College"
    },
    "LNMIIT Jaipur": {
        "established": 2002, "type": "Private / Deemed", "city": "Jaipur, Rajasthan",
        "total_students": 4500, "faculty": 220, "nirf_rank": 85, "naac": "A",
        "avg_package_lpa": 10, "highest_package_lpa": 50,
        "placement_pct": 83, "top_recruiters": "TCS, Infosys, Wipro, Flipkart, Amazon",
        "fee_per_year_lpa": 2.5, "campus_acres": 100,
        "known_for": "LNM group, Jaipur IT growth zone, good startup ecosystem",
        "roi_score": 76, "roi_rationale": "Moderate ROI — Jaipur growth helps but fees are high vs packages",
        "hostel": "Available", "sports": "Good facilities",
        "research_centers": 7, "international_collab": "Rajasthan tech corridor"
    },
    "Jaypee Solan": {
        "established": 2002, "type": "Private / Deemed", "city": "Solan, HP",
        "total_students": 3500, "faculty": 180, "nirf_rank": 100, "naac": "A",
        "avg_package_lpa": 8.5, "highest_package_lpa": 38,
        "placement_pct": 78, "top_recruiters": "Infosys, TCS, Wipro, IBM, Accenture",
        "fee_per_year_lpa": 3.0, "campus_acres": 200,
        "known_for": "Himachal hill campus, IT focus, pharma industry ties",
        "roi_score": 68, "roi_rationale": "Higher fees vs moderate packages — ROI depends on branch chosen",
        "hostel": "Available (scenic hill station)", "sports": "Outdoor hill sports",
        "research_centers": 5, "international_collab": "Jaypee group industries"
    },
    # ── Medical ───────────────────────────────────────────────────────────────
    "AIIMS Delhi": {
        "established": 1956, "type": "Central / AIIMS", "city": "New Delhi",
        "total_students": 3500, "faculty": 1200, "nirf_rank": 1, "naac": "A++",
        "avg_package_lpa": 25, "highest_package_lpa": 80,
        "placement_pct": 100, "top_recruiters": "Govt hospitals, WHO, ICMR, Private practice",
        "fee_per_year_lpa": 0.01, "campus_acres": 195,
        "known_for": "India's #1 medical college, MBBS fee is Rs.1628/yr — world's best value",
        "roi_score": 100, "roi_rationale": "MBBS at Rs.1628/year + 100% placement = greatest ROI in Indian education",
        "hostel": "Mandatory & free", "sports": "Full AIIMS campus facilities",
        "research_centers": 40, "international_collab": "WHO, Johns Hopkins, Harvard Medical"
    },
    "MAMC Delhi": {
        "established": 1958, "type": "State / Delhi Govt", "city": "New Delhi",
        "total_students": 2000, "faculty": 500, "nirf_rank": 6, "naac": "A+",
        "avg_package_lpa": 18, "highest_package_lpa": 60,
        "placement_pct": 98, "top_recruiters": "Delhi Govt hospitals, AIIMS referrals, Private",
        "fee_per_year_lpa": 0.05, "campus_acres": 120,
        "known_for": "Maulana Azad Medical — Delhi's #1 state med college, LNJP hospital attachment",
        "roi_score": 99, "roi_rationale": "Minimal fees + Delhi location + top placements = extraordinary ROI",
        "hostel": "Available", "sports": "Delhi city access",
        "research_centers": 20, "international_collab": "Delhi Health Ministry programs"
    },
    "VMMC Delhi": {
        "established": 1954, "type": "State / Delhi Govt", "city": "New Delhi",
        "total_students": 1800, "faculty": 450, "nirf_rank": 8, "naac": "A+",
        "avg_package_lpa": 17, "highest_package_lpa": 55,
        "placement_pct": 97, "top_recruiters": "Safdarjung Hospital, Private hospitals Delhi",
        "fee_per_year_lpa": 0.05, "campus_acres": 100,
        "known_for": "Vardhmaan Mahavir — attached to Safdarjung hospital (India's largest hospital)",
        "roi_score": 98, "roi_rationale": "Free education + Safdarjung exposure = top-tier career ROI",
        "hostel": "Available", "sports": "Campus facilities",
        "research_centers": 15, "international_collab": "WHO projects"
    },
    "AIIMS Rishikesh": {
        "established": 2012, "type": "Central / AIIMS", "city": "Rishikesh, Uttarakhand",
        "total_students": 2200, "faculty": 600, "nirf_rank": 14, "naac": "A+",
        "avg_package_lpa": 20, "highest_package_lpa": 65,
        "placement_pct": 99, "top_recruiters": "Central Govt, State Govt hospitals, WHO",
        "fee_per_year_lpa": 0.01, "campus_acres": 180,
        "known_for": "Second AIIMS of India, Uttarakhand healthcare hub",
        "roi_score": 99, "roi_rationale": "AIIMS brand + near-zero fees = near-perfect ROI",
        "hostel": "Mandatory & subsidised", "sports": "River Ganga nearby, full campus",
        "research_centers": 25, "international_collab": "AIIMS consortium"
    },
    "KGMU Lucknow": {
        "established": 1905, "type": "State / UP Govt", "city": "Lucknow, UP",
        "total_students": 5000, "faculty": 800, "nirf_rank": 11, "naac": "A+",
        "avg_package_lpa": 15, "highest_package_lpa": 50,
        "placement_pct": 96, "top_recruiters": "UP Govt health dept, private hospitals, PGIMER",
        "fee_per_year_lpa": 0.08, "campus_acres": 200,
        "known_for": "King George — one of oldest medical schools in Asia, 700-bed teaching hospital",
        "roi_score": 97, "roi_rationale": "Heritage + very low fees + UP public sector = excellent career ROI",
        "hostel": "Available", "sports": "Full Lucknow campus",
        "research_centers": 22, "international_collab": "British era heritage linkages"
    },
    "Thapar University": {
        "established": 1956, "type": "Private / Deemed", "city": "Patiala, Punjab",
        "total_students": 12000, "faculty": 600, "nirf_rank": 27, "naac": "A+",
        "avg_package_lpa": 14, "highest_package_lpa": 80,
        "placement_pct": 91, "top_recruiters": "Google, Microsoft, Nvidia, Goldman Sachs, Samsung",
        "fee_per_year_lpa": 3.5, "campus_acres": 250,
        "known_for": "Best private engineering college in North India, Silicon Valley network",
        "roi_score": 85, "roi_rationale": "Higher fees but top packages — strong ROI especially for CS",
        "hostel": "Available (world-class)", "sports": "Olympic-standard facilities",
        "research_centers": 20, "international_collab": "UBC Canada, TU Delft"
    },
    # ── Placeholder for colleges without detail entries ───────────────────────
}

def get_college_detail(name: str) -> dict:
    """Return college details or a sensible default."""
    if name in COLLEGE_DETAILS:
        return COLLEGE_DETAILS[name]
    return {
        "established": "N/A", "type": "N/A", "city": "North India",
        "total_students": "N/A", "faculty": "N/A", "nirf_rank": "N/A", "naac": "N/A",
        "avg_package_lpa": "N/A", "highest_package_lpa": "N/A",
        "placement_pct": "N/A", "top_recruiters": "To be updated",
        "fee_per_year_lpa": "N/A", "campus_acres": "N/A",
        "known_for": "Data being compiled for this institution",
        "roi_score": 70, "roi_rationale": "Insufficient data — estimate only",
        "hostel": "Contact college", "sports": "Contact college",
        "research_centers": "N/A", "international_collab": "N/A"
    }

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class EngineeringInput:
    jee_main_percentile: float
    jee_advanced_rank: int
    class_12_percent: float
    category: str
    state: str
    branch: str

@dataclass
class MedicalInput:
    neet_score: int
    neet_rank: int
    class_12_percent: float
    category: str
    domicile: str
    course: str

@dataclass
class PredictionResult:
    college: str
    location: str
    type: str
    program: str
    ml_probability: float
    rules_score: float
    final_score: float
    category: str
    rationale: str

# ==============================================================================
# ML MODEL LOADER
# ==============================================================================

@st.cache_resource(show_spinner="Loading ML Models…")
def load_ml_models() -> Tuple[Any, Any, Optional[str]]:
    engineering_model = None
    category_encoder  = None
    error_msg         = None
    try:
        current_dir  = os.path.dirname(os.path.abspath(__file__))
        model_path   = os.path.join(current_dir, 'engineering_model_north.pkl')
        encoder_path = os.path.join(current_dir, 'category_encoder_north.pkl')
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            engineering_model = joblib.load(model_path, mmap_mode='r')
            category_encoder  = joblib.load(encoder_path)
        else:
            error_msg = "ML Model files not found. Using rule-based fallback."
    except Exception as e:
        error_msg = f"Error loading ML models: {str(e)}"
    return engineering_model, category_encoder, error_msg


class DataPreprocessor:
    @staticmethod
    def preprocess_engineering(data: EngineeringInput, encoder) -> np.ndarray:
        rank = data.jee_advanced_rank
        if rank == 0 or rank >= 999999:
            rank = int((1 - data.jee_main_percentile / 100) * 1400000)
        category_encoded = CATEGORY_MAP.get(data.category, 0)
        if encoder is not None:
            try:
                category_encoded = encoder.transform([data.category])[0]
            except Exception:
                pass
        return np.array([[rank, category_encoded]])


class MLPredictor:
    @staticmethod
    def predict_engineering(features: np.ndarray, model) -> Dict[str, Any]:
        if model is None:
            return {'success': False, 'error': 'Model not loaded'}
        try:
            if hasattr(model, "predict_proba"):
                probabilities   = model.predict_proba(features)[0]
                max_prob_index  = np.argmax(probabilities)
                predicted_class = model.classes_[max_prob_index]
                confidence      = float(probabilities[max_prob_index]) * 100
            else:
                predicted_class = model.predict(features)[0]
                confidence      = 85.0
            prediction_str = str(predicted_class)
            if " - " in prediction_str:
                college, branch = prediction_str.split(" - ", 1)
            else:
                college, branch = prediction_str, "Engineering"
            return {'success': True, 'college': college, 'branch': branch, 'confidence': confidence}
        except Exception as e:
            return {'success': False, 'error': str(e)}

# ==============================================================================
# RULES ENGINE
# ==============================================================================

class RulesEngine:
    @staticmethod
    def evaluate_engineering(data: EngineeringInput) -> Dict[str, Any]:
        if data.class_12_percent < 75 and data.category == 'General':
            return {'eligible': False,
                    'message': 'Class 12 >= 75% required for General category in premier institutes.',
                    'candidates': []}

        mains_rank = int((1 - data.jee_main_percentile / 100) * 1400000) if data.jee_main_percentile > 0 else 999999
        adv_rank   = data.jee_advanced_rank if data.jee_advanced_rank > 0 else 999999

        cat_mult    = {'General': 1.0, 'OBC-NCL': 1.2, 'EWS': 1.15, 'SC': 3.5, 'ST': 4.5}
        branch_mult = {'Computer Science': 1.0, 'Electronics': 1.5, 'Electrical': 2.0, 'Mechanical': 3.0}
        total_multiplier = cat_mult.get(data.category, 1.0) * branch_mult.get(data.branch, 1.0)

        eng_db = [
            {"name": "IIT Delhi",         "state": "Delhi",              "type": "[GOVT - IIT]",      "exam": "adv",   "cutoff": 120,   "hs_quota": False},
            {"name": "IIT Kanpur",         "state": "Uttar Pradesh",      "type": "[GOVT - IIT]",      "exam": "adv",   "cutoff": 250,   "hs_quota": False},
            {"name": "IIT Roorkee",        "state": "Uttarakhand",        "type": "[GOVT - IIT]",      "exam": "adv",   "cutoff": 450,   "hs_quota": False},
            {"name": "IIT BHU Varanasi",   "state": "Uttar Pradesh",      "type": "[GOVT - IIT]",      "exam": "adv",   "cutoff": 900,   "hs_quota": False},
            {"name": "IIT Ropar",          "state": "Punjab",             "type": "[GOVT - IIT]",      "exam": "adv",   "cutoff": 1600,  "hs_quota": False},
            {"name": "IIT Jodhpur",        "state": "Rajasthan",          "type": "[GOVT - IIT]",      "exam": "adv",   "cutoff": 2200,  "hs_quota": False},
            {"name": "IIT Mandi",          "state": "Himachal Pradesh",   "type": "[GOVT - IIT]",      "exam": "adv",   "cutoff": 2600,  "hs_quota": False},
            {"name": "IIT Jammu",          "state": "Jammu & Kashmir",    "type": "[GOVT - IIT]",      "exam": "adv",   "cutoff": 4800,  "hs_quota": False},
            {"name": "MNNIT Allahabad",    "state": "Uttar Pradesh",      "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 4500,  "hs_quota": True, "hs_mult": 1.8},
            {"name": "MNIT Jaipur",        "state": "Rajasthan",          "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 5500,  "hs_quota": True, "hs_mult": 1.7},
            {"name": "NIT Kurukshetra",    "state": "Haryana",            "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 7500,  "hs_quota": True, "hs_mult": 1.9},
            {"name": "IIIT Allahabad",     "state": "Uttar Pradesh",      "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 5500,  "hs_quota": False},
            {"name": "NIT Jalandhar",      "state": "Punjab",             "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 12500, "hs_quota": True, "hs_mult": 2.2},
            {"name": "NIT Delhi",          "state": "Delhi",              "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 8500,  "hs_quota": True, "hs_mult": 2.5},
            {"name": "NIT Hamirpur",       "state": "Himachal Pradesh",   "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 16000, "hs_quota": True, "hs_mult": 2.8},
            {"name": "NIT Uttarakhand",    "state": "Uttarakhand",        "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 21000, "hs_quota": True, "hs_mult": 2.5},
            {"name": "NIT Srinagar",       "state": "Jammu & Kashmir",    "type": "[GOVT - NIT/IIIT]", "exam": "mains", "cutoff": 28000, "hs_quota": True, "hs_mult": 3.0},
            {"name": "DTU Delhi",          "state": "Delhi",              "type": "[GOVT - STATE]",    "exam": "mains", "cutoff": 6500,  "hs_quota": True, "hs_mult": 3.5},
            {"name": "NSUT Delhi",         "state": "Delhi",              "type": "[GOVT - STATE]",    "exam": "mains", "cutoff": 7500,  "hs_quota": True, "hs_mult": 3.5},
            {"name": "IGDTUW Delhi (Women)","state": "Delhi",             "type": "[GOVT - STATE]",    "exam": "mains", "cutoff": 13000, "hs_quota": True, "hs_mult": 3.0},
            {"name": "PEC Chandigarh",     "state": "Punjab",             "type": "[GOVT - STATE]",    "exam": "mains", "cutoff": 11000, "hs_quota": True, "hs_mult": 2.0},
            {"name": "HBTU Kanpur",        "state": "Uttar Pradesh",      "type": "[GOVT - STATE]",    "exam": "mains", "cutoff": 18000, "hs_quota": True, "hs_mult": 2.5},
            {"name": "Thapar University",  "state": "Punjab",             "type": "[PVT - TOP TIER]",  "exam": "mains", "cutoff": 40000, "hs_quota": True, "hs_mult": 1.5},
            {"name": "JIIT Noida",         "state": "Uttar Pradesh",      "type": "[PVT - TOP TIER]",  "exam": "mains", "cutoff": 55000, "hs_quota": False},
            {"name": "Shiv Nadar University","state": "Uttar Pradesh",    "type": "[PVT - TOP TIER]",  "exam": "mains", "cutoff": 48000, "hs_quota": False},
            {"name": "LNMIIT Jaipur",      "state": "Rajasthan",          "type": "[PVT - TOP TIER]",  "exam": "mains", "cutoff": 32000, "hs_quota": False},
            {"name": "Jaypee Solan",       "state": "Himachal Pradesh",   "type": "[PVT - TOP TIER]",  "exam": "mains", "cutoff": 85000, "hs_quota": False},
        ]

        candidates = []
        for college in eng_db:
            if college['state'] not in NORTH_STATES:
                continue
            is_hs = (data.state == college['state'])
            if college['exam'] == 'adv':
                if adv_rank == 999999: continue
                user_rank = adv_rank
            else:
                user_rank = mains_rank
            base_cutoff = college['cutoff']
            if college['hs_quota'] and is_hs:
                base_cutoff = int(base_cutoff * college.get('hs_mult', 2.0))
            final_cutoff = int(base_cutoff * total_multiplier)
            match_category, rules_score = None, 0.0
            if user_rank <= final_cutoff * 0.85:
                match_category = "high-chance"
                rules_score    = min(99.0, ((final_cutoff / max(1, user_rank)) * 40) + 50)
            elif user_rank <= final_cutoff * 1.20:
                match_category = "moderate-borderline"
                rules_score    = 75.0 - (((user_rank - final_cutoff * 0.85) / max(1, final_cutoff * 0.35)) * 15)
            elif user_rank <= final_cutoff * 2.5:
                match_category = "dream-colleges"
                rules_score    = 60.0 - (((user_rank - final_cutoff * 1.20) / max(1, final_cutoff * 1.3)) * 20)
            if match_category:
                quota_str  = " (Home State Quota)" if (is_hs and college['hs_quota']) else " (Other State / AIQ)"
                eval_method = "JEE Advanced Rank" if college['exam'] == 'adv' else f"JEE Mains AIR (~{mains_rank})"
                candidates.append({
                    'college': college['name'], 'location': college['state'],
                    'type': college['type'], 'branch': data.branch,
                    'rules_score': max(0.0, round(rules_score, 1)),
                    'match_category': match_category,
                    'rationale': f"Evaluated via {eval_method}{quota_str} against historical trends."
                })
        return {'eligible': True, 'candidates': candidates}

    @staticmethod
    def evaluate_medical(data: MedicalInput) -> Dict[str, Any]:
        if data.class_12_percent < 50 and data.category == 'General':
            return {'eligible': False, 'message': 'Class 12 >= 50% required.', 'candidates': []}
        candidates = []
        user_rank  = data.neet_rank if data.neet_rank > 0 else 999999
        cat_mult   = {'General': 1.0, 'OBC-NCL': 1.1, 'EWS': 1.15, 'SC': 3.5, 'ST': 4.5}
        course_mult = 1.5 if data.course == 'BDS' else 1.0
        total_mult  = cat_mult.get(data.category, 1.0) * course_mult
        medical_db  = [
            {"name": "AIIMS Delhi",         "state": "Delhi",            "type": "Central",    "cutoff": 57},
            {"name": "MAMC Delhi",          "state": "Delhi",            "type": "State Govt", "cutoff": 85},
            {"name": "VMMC Delhi",          "state": "Delhi",            "type": "State Govt", "cutoff": 107},
            {"name": "RML Delhi",           "state": "Delhi",            "type": "State Govt", "cutoff": 190},
            {"name": "UCMS Delhi",          "state": "Delhi",            "type": "State Govt", "cutoff": 300},
            {"name": "AIIMS Jodhpur",       "state": "Rajasthan",        "type": "Central",    "cutoff": 497},
            {"name": "IMS BHU",             "state": "Uttar Pradesh",    "type": "Central",    "cutoff": 858},
            {"name": "AIIMS Rishikesh",     "state": "Uttarakhand",      "type": "Central",    "cutoff": 931},
            {"name": "KGMU Lucknow",        "state": "Uttar Pradesh",    "type": "State Govt", "cutoff": 1200},
            {"name": "AIIMS Bathinda",      "state": "Punjab",           "type": "Central",    "cutoff": 1900},
            {"name": "SMS Jaipur",          "state": "Rajasthan",        "type": "State Govt", "cutoff": 2200},
            {"name": "AIIMS Gorakhpur",     "state": "Uttar Pradesh",    "type": "Central",    "cutoff": 2600},
            {"name": "AIIMS Raebareli",     "state": "Uttar Pradesh",    "type": "Central",    "cutoff": 3000},
            {"name": "AIIMS Jammu",         "state": "Jammu & Kashmir",  "type": "Central",    "cutoff": 4000},
            {"name": "Pt. BDS Rohtak",      "state": "Haryana",          "type": "State Govt", "cutoff": 4500},
            {"name": "IGMC Shimla",         "state": "Himachal Pradesh", "type": "State Govt", "cutoff": 5000},
            {"name": "GMC Patiala",         "state": "Punjab",           "type": "State Govt", "cutoff": 6500},
            {"name": "SNMC Agra",           "state": "Uttar Pradesh",    "type": "State Govt", "cutoff": 7500},
            {"name": "Doon Medical College","state": "Uttarakhand",      "type": "State Govt", "cutoff": 8500},
            {"name": "RNT Medical College", "state": "Rajasthan",        "type": "State Govt", "cutoff": 9000},
            {"name": "GMC Haldwani",        "state": "Uttarakhand",      "type": "State Govt", "cutoff": 10500},
            {"name": "KCGMC Karnal",        "state": "Haryana",          "type": "State Govt", "cutoff": 12000},
            {"name": "GMC Amritsar",        "state": "Punjab",           "type": "State Govt", "cutoff": 14000},
            {"name": "RPGMC Tanda",         "state": "Himachal Pradesh", "type": "State Govt", "cutoff": 15500},
            {"name": "ASMC Basti",          "state": "Uttar Pradesh",    "type": "State Govt", "cutoff": 18500},
            {"name": "GMC Srinagar",        "state": "Jammu & Kashmir",  "type": "State Govt", "cutoff": 21000},
            {"name": "GMC Jammu",           "state": "Jammu & Kashmir",  "type": "State Govt", "cutoff": 22000},
            {"name": "Sharda University",   "state": "Uttar Pradesh",    "type": "Private",    "cutoff": 250000},
            {"name": "SGT University",      "state": "Haryana",          "type": "Private",    "cutoff": 350000},
            {"name": "JNUIMSRC Jaipur",     "state": "Rajasthan",        "type": "Private",    "cutoff": 450000},
            {"name": "SGRR Dehradun",       "state": "Uttarakhand",      "type": "Private",    "cutoff": 500000},
            {"name": "Adesh Medical College","state": "Punjab",           "type": "Private",    "cutoff": 600000},
            {"name": "Maharishi Markandeshwar","state": "Haryana",       "type": "Deemed",     "cutoff": 650000},
            {"name": "Santosh Medical College","state": "Uttar Pradesh",  "type": "Deemed",     "cutoff": 800000},
        ]
        for college in medical_db:
            if college['state'] not in NORTH_STATES: continue
            base_cutoff = college['cutoff']
            if data.domicile == college['state']:
                base_cutoff = int(base_cutoff * 1.5)
            final_cutoff = int(base_cutoff * total_mult)
            match_category, rules_score = None, 0.0
            if user_rank <= final_cutoff * 0.85:
                match_category = "high-chance"
                rules_score    = min(99.0, ((final_cutoff / max(1, user_rank)) * 40) + 50)
            elif user_rank <= final_cutoff * 1.20:
                match_category = "moderate-borderline"
                rules_score    = 75.0 - (((user_rank - final_cutoff * 0.85) / max(1, final_cutoff * 0.35)) * 15)
            elif user_rank <= final_cutoff * 2.5:
                match_category = "dream-colleges"
                rules_score    = 60.0 - (((user_rank - final_cutoff * 1.20) / max(1, final_cutoff * 1.3)) * 20)
            if match_category:
                candidates.append({
                    'college': college['name'], 'location': college['state'],
                    'type': college['type'], 'course': data.course,
                    'rules_score': max(0.0, round(rules_score, 1)),
                    'match_category': match_category
                })
        return {'eligible': True, 'candidates': candidates}


class RankingEngine:
    @staticmethod
    def format_engineering_results(ml_result, rules_result):
        if not rules_result.get('eligible', True): return rules_result
        final_rankings, seen = [], set()
        for cand in rules_result.get('candidates', []):
            ml_prob   = 70.0
            rationale = cand.get('rationale', '')
            if ml_result.get('success') and ml_result['college'] in cand['college']:
                ml_prob = ml_result['confidence']
                seen.add(ml_result['college'])
                rationale += " ML pattern confirmed."
            rules_score = cand['rules_score']
            final_score = (0.6 * ml_prob) + (0.4 * rules_score)
            final_rankings.append(PredictionResult(
                college=cand['college'], location=cand['location'], type=cand['type'],
                program=cand['branch'], ml_probability=ml_prob, rules_score=rules_score,
                final_score=final_score, category=cand['match_category'], rationale=rationale
            ))
        if ml_result.get('success') and ml_result['college'] not in seen:
            ml_prob, rules_score = ml_result['confidence'], 60.0
            final_score = (0.6 * ml_prob) + (0.4 * rules_score)
            category = "high-chance" if final_score >= 85 else "moderate-borderline" if final_score >= 75 else "dream-colleges"
            final_rankings.append(PredictionResult(
                college=ml_result['college'], location="North India", type="[AI PREDICTED]",
                program=ml_result['branch'], ml_probability=ml_prob, rules_score=rules_score,
                final_score=final_score, category=category,
                rationale="Identified via Machine Learning pattern matching."
            ))
        final_rankings.sort(key=lambda x: x.final_score, reverse=True)
        top_score = final_rankings[0].final_score if final_rankings else 0.0
        return {'eligible': True, 'overall_score': top_score, 'rankings': final_rankings}

    @staticmethod
    def format_medical_results(rules_result):
        if not rules_result.get('eligible', True): return rules_result
        final_rankings = [
            PredictionResult(
                college=c['college'], location=c['location'], type=c['type'],
                program=c['course'], ml_probability=0.0, rules_score=c['rules_score'],
                final_score=c['rules_score'], category=c['match_category'],
                rationale="Matched via exhaustive AIR database scan."
            ) for c in rules_result.get('candidates', [])
        ]
        final_rankings.sort(key=lambda x: x.final_score, reverse=True)
        top_score = final_rankings[0].final_score if final_rankings else 0.0
        return {'eligible': True, 'overall_score': top_score, 'rankings': final_rankings}

# ==============================================================================
# SESSION STATE HELPERS
# ==============================================================================

def save_to_dashboard(results: dict, stream: str, inputs: dict):
    if 'dashboard_history' not in st.session_state:
        st.session_state.dashboard_history = []
    entry = {
        'stream': stream,
        'inputs': inputs,
        'overall_score': results.get('overall_score', 0),
        'top_college': results['rankings'][0].college if results.get('rankings') else 'N/A',
        'top_category': results['rankings'][0].category if results.get('rankings') else 'N/A',
        'rankings': results.get('rankings', []),
        'timestamp': pd.Timestamp.now().strftime("%d %b %Y, %I:%M %p")
    }
    st.session_state.dashboard_history.insert(0, entry)
    if len(st.session_state.dashboard_history) > 10:
        st.session_state.dashboard_history = st.session_state.dashboard_history[:10]

# ==============================================================================
# CSS
# ==============================================================================

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700;900&family=Public+Sans:wght@400;600;800&display=swap');

    * { font-family: 'Public Sans', sans-serif; }
    .stApp { background-color: #f4f0ec; background-image: radial-gradient(#00000018 1px, transparent 1px); background-size: 22px 22px; }
    .main .block-container { background: #fff; border: 4px solid #000; padding: 2.5rem 3rem; box-shadow: 10px 10px 0px #000; margin-top: 1.5rem; margin-bottom: 2rem; border-radius: 0; }

    /* NAV */
    .nav-bar { display:flex; gap:1rem; align-items:center; justify-content:center; margin-bottom:2rem; flex-wrap:wrap; }
    .nav-btn { background:#fff; border:3px solid #000; padding:0.5rem 1.4rem; font-family:'Space Grotesk',sans-serif; font-weight:800; font-size:0.95rem; cursor:pointer; box-shadow:4px 4px 0 #000; text-transform:uppercase; text-decoration:none; color:#000; transition:all .12s; }
    .nav-btn:hover { transform:translate(-2px,-2px); box-shadow:6px 6px 0 #000; background:#00f5d4; }
    .nav-btn.active { background:#000; color:#fff; }

    /* TITLE */
    .main-title { font-family:'Space Grotesk',sans-serif; font-size:3rem; font-weight:900; text-align:center; color:#000; text-transform:uppercase; padding-bottom:1rem; }
    .subtitle-pill { display:inline-block; background:#ffd166; border:3px solid #000; padding:.35rem 1.2rem; font-family:'Space Grotesk',sans-serif; font-weight:800; font-size:1rem; text-transform:uppercase; }
    .region-badge { display:inline-flex; align-items:center; gap:5px; background:#fff; border:3px solid #000; box-shadow:4px 4px 0 #000; padding:.45rem 1rem; font-weight:800; font-size:.85rem; text-transform:uppercase; }

    /* STREAM CARDS */
    .stream-grid { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin:1.5rem 0; }
    .stream-card { background:#fff; border:4px solid #000; box-shadow:8px 8px 0 #000; padding:2.5rem 1.5rem; text-align:center; cursor:pointer; transition:all .15s; position:relative; }
    .stream-card:hover { transform:translate(-3px,-3px); box-shadow:11px 11px 0 #000; background:#cce3de; }
    .stream-card.coming-soon { background:#f9f9f9; cursor:default; opacity:.85; }
    .stream-card.coming-soon:hover { transform:none; box-shadow:8px 8px 0 #000; background:#f9f9f9; }
    .cs-badge { position:absolute; top:12px; right:12px; background:#ff6b6b; color:#fff; border:2px solid #000; padding:.2rem .6rem; font-size:.75rem; font-weight:900; font-family:'Space Grotesk',sans-serif; text-transform:uppercase; }
    .stream-icon { font-size:3.5rem; margin-bottom:.8rem; }
    .stream-title { font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:900; text-transform:uppercase; }
    .stream-desc { font-size:.9rem; font-weight:600; color:#333; margin-top:.4rem; line-height:1.5; }

    /* FORM */
    .form-section-title { font-family:'Space Grotesk',sans-serif; font-size:1.3rem; font-weight:900; background:#ffcfd2; border:3px solid #000; padding:.4rem 1rem; display:inline-block; text-transform:uppercase; margin-bottom:1rem; }

    /* RESULTS */
    .metric-card { background:#ffb703; border:4px solid #000; box-shadow:8px 8px 0 #000; text-align:center; padding:2rem; margin:1rem 0; }
    .metric-value { font-family:'Space Grotesk',sans-serif; font-size:3.5rem; font-weight:900; }
    .metric-label { font-size:1rem; font-weight:800; text-transform:uppercase; margin-top:.3rem; }

    .college-card { background:#fff; border:4px solid #000; box-shadow:8px 8px 0 #000; padding:1.8rem; margin:1.5rem 0; position:relative; }
    .college-card.high-chance { background:#b9fbc0; }
    .college-card.moderate-borderline { background:#fbf8cc; }
    .college-card.dream-colleges { background:#ffcfd2; }
    .college-name { font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:900; text-transform:uppercase; margin-bottom:.3rem; }
    .college-type-badge { display:inline-block; background:#000; color:#fff; padding:.3rem .8rem; font-size:.78rem; font-weight:700; text-transform:uppercase; margin-right:.5rem; margin-bottom:.6rem; }
    .tier-label { font-family:'Space Grotesk',sans-serif; font-weight:900; font-size:.9rem; margin-bottom:.3rem; }
    .rank-badge { position:absolute; top:1.2rem; right:1.2rem; background:#000; color:#fff; width:44px; height:44px; display:flex; align-items:center; justify-content:center; font-family:'Space Grotesk',sans-serif; font-weight:900; font-size:1.1rem; }
    .custom-progress { width:100%; height:18px; background:#fff; border:3px solid #000; margin:1rem 0; }
    .custom-progress-bar { height:100%; background:#000; }
    .roi-bar { width:100%; height:14px; background:#f0f0f0; border:2px solid #000; margin:.4rem 0; }
    .roi-fill { height:100%; }

    /* POPUP OVERLAY */
    .popup-overlay { position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.55); z-index:9998; display:flex; align-items:center; justify-content:center; }
    .popup-box { background:#fff; border:4px solid #000; box-shadow:12px 12px 0 #000; padding:2.5rem; max-width:720px; width:95%; max-height:85vh; overflow-y:auto; position:relative; z-index:9999; }
    .popup-title { font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:900; text-transform:uppercase; border-bottom:4px solid #000; padding-bottom:.8rem; margin-bottom:1.2rem; }
    .detail-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1.2rem; }
    .detail-item { border:3px solid #000; padding:.7rem 1rem; background:#f9f9f9; }
    .detail-label { font-size:.75rem; font-weight:800; text-transform:uppercase; color:#555; margin-bottom:.2rem; }
    .detail-value { font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:900; }
    .roi-section { background:#ffd166; border:3px solid #000; padding:1rem; margin-bottom:1rem; }

    /* BUTTON */
    .stButton>button { width:100%; background:#00f5d4; color:#000; border:4px solid #000; box-shadow:6px 6px 0 #000; font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:900; padding:.9rem; text-transform:uppercase; transition:all .1s; border-radius:0; }
    .stButton>button:hover { transform:translate(-2px,-2px); box-shadow:8px 8px 0 #000; color:#000; }
    .stButton>button:active { transform:translate(3px,3px); box-shadow:0 0 0 #000; }

    /* INFO BOX */
    .info-box { border:4px solid #000; box-shadow:6px 6px 0 #000; padding:1.2rem 1.5rem; margin:1.2rem 0; background:#fff; }
    .info-box.warning { background:#ffb703; }
    .info-box.advisory { background:#e0f7fa; border-color:#00838f; }
    .info-box-title { font-family:'Space Grotesk',sans-serif; font-weight:900; font-size:1.1rem; text-transform:uppercase; margin-bottom:.3rem; }

    /* INPUTS */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div { border:3px solid #000 !important; border-radius:0 !important; font-weight:600 !important; }
    #MainMenu,footer,header { visibility:hidden; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# COLLEGE DETAIL POPUP
# ==============================================================================

def _fmt(val) -> str:
    """Safely format a number with commas, or return the value as-is if not numeric."""
    try:
        return f"{int(val):,}"
    except (ValueError, TypeError):
        return str(val)

def render_college_popup(college_name: str, stream: str = "Engineering"):
    """Renders a popup modal for college details. Hides placement stats for Medical stream."""
    d = get_college_detail(college_name)
    roi = d.get('roi_score', 70)
    roi_color = "#06d6a0" if roi >= 90 else "#ffb703" if roi >= 75 else "#ef476f"
    roi_medal = "🥇" if roi >= 90 else "🥈" if roi >= 75 else "🥉"
    is_medical = (stream == 'Medical')

    with st.container():
        st.markdown(f"""
        <div style="background:#fff;border:4px solid #000;box-shadow:12px 12px 0 #000;padding:2rem;margin:1rem 0;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:900;text-transform:uppercase;
               border-bottom:4px solid #000;padding-bottom:.7rem;margin-bottom:1.2rem;">
            🏛 {college_name}
          </div>

          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem;margin-bottom:1rem;">
            <div style="border:3px solid #000;padding:.6rem .9rem;background:#f0f4f8;">
              <div style="font-size:.7rem;font-weight:800;text-transform:uppercase;color:#555;">Established</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:900;">{d['established']}</div>
            </div>
            <div style="border:3px solid #000;padding:.6rem .9rem;background:#f0f4f8;">
              <div style="font-size:.7rem;font-weight:800;text-transform:uppercase;color:#555;">City</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:900;">📍 {d['city']}</div>
            </div>
            <div style="border:3px solid #000;padding:.6rem .9rem;background:#f0f4f8;">
              <div style="font-size:.7rem;font-weight:800;text-transform:uppercase;color:#555;">Type</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:900;">{d['type']}</div>
            </div>
            <div style="border:3px solid #000;padding:.6rem .9rem;background:#b9fbc0;">
              <div style="font-size:.7rem;font-weight:800;text-transform:uppercase;color:#555;">NIRF Rank</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:900;">#{d['nirf_rank']}</div>
            </div>
            <div style="border:3px solid #000;padding:.6rem .9rem;background:#b9fbc0;">
              <div style="font-size:.7rem;font-weight:800;text-transform:uppercase;color:#555;">NAAC Grade</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:900;">{d['naac']}</div>
            </div>
            <div style="border:3px solid #000;padding:.6rem .9rem;background:#b9fbc0;">
              <div style="font-size:.7rem;font-weight:800;text-transform:uppercase;color:#555;">Campus</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:900;">{d['campus_acres']} acres</div>
            </div>
          </div>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-bottom:1rem;">
            <div style="border:3px solid #000;padding:.6rem .9rem;background:#fbf8cc;">
              <div style="font-size:.7rem;font-weight:800;text-transform:uppercase;color:#555;">Total Students</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:900;">{_fmt(d['total_students'])}</div>
            </div>
            <div style="border:3px solid #000;padding:.6rem .9rem;background:#fbf8cc;">
              <div style="font-size:.7rem;font-weight:800;text-transform:uppercase;color:#555;">Faculty</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:900;">{_fmt(d['faculty'])}</div>
            </div>
            <div style="border:3px solid #000;padding:.6rem .9rem;background:#fbf8cc;">
              <div style="font-size:.7rem;font-weight:800;text-transform:uppercase;color:#555;">Research Centers</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:900;">{d['research_centers']}</div>
            </div>
            <div style="border:3px solid #000;padding:.6rem .9rem;background:#fbf8cc;">
              <div style="font-size:.7rem;font-weight:800;text-transform:uppercase;color:#555;">International Collab</div>
              <div style="font-size:.9rem;font-weight:700;">{d['international_collab']}</div>
            </div>
          </div>

          {"" if is_medical else f"""
          <div style="border:3px solid #000;padding:1rem;background:#cce3de;margin-bottom:1rem;">
            <div style="font-size:.75rem;font-weight:800;text-transform:uppercase;color:#333;margin-bottom:.5rem;">📦 Placement Stats</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.5rem;">
              <div>
                <div style="font-size:.7rem;font-weight:700;color:#555;">Avg Package</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:900;color:#0d6e47;">₹{d['avg_package_lpa']} LPA</div>
              </div>
              <div>
                <div style="font-size:.7rem;font-weight:700;color:#555;">Highest Package</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:900;color:#0d6e47;">₹{d['highest_package_lpa']} LPA</div>
              </div>
              <div>
                <div style="font-size:.7rem;font-weight:700;color:#555;">Placement %</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:900;color:#0d6e47;">{d['placement_pct']}%</div>
              </div>
            </div>
            <div style="margin-top:.6rem;font-size:.85rem;font-weight:700;">🏢 Top Recruiters: {d['top_recruiters']}</div>
          </div>"""}

          <div style="border:3px solid #000;padding:1rem;background:#ffd166;margin-bottom:1rem;">
            <div style="font-size:.75rem;font-weight:800;text-transform:uppercase;color:#333;margin-bottom:.5rem;">💰 ROI Analysis {roi_medal}</div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem;">
              <span style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:900;">ROI Score: {roi}/100</span>
              <span style="font-size:.85rem;font-weight:700;">Fee/yr: ₹{d['fee_per_year_lpa']} LPA</span>
            </div>
            <div style="width:100%;height:16px;background:#fff;border:2px solid #000;margin:.5rem 0;">
              <div style="height:100%;width:{roi}%;background:{roi_color};"></div>
            </div>
            <div style="font-size:.88rem;font-weight:700;color:#333;">💡 {d['roi_rationale']}</div>
          </div>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-bottom:1rem;">
            <div style="border:3px solid #000;padding:.6rem .9rem;background:#f9f9f9;">
              <div style="font-size:.7rem;font-weight:800;text-transform:uppercase;color:#555;">Hostel</div>
              <div style="font-size:.9rem;font-weight:700;">🏠 {d['hostel']}</div>
            </div>
            <div style="border:3px solid #000;padding:.6rem .9rem;background:#f9f9f9;">
              <div style="font-size:.7rem;font-weight:800;text-transform:uppercase;color:#555;">Sports</div>
              <div style="font-size:.9rem;font-weight:700;">⚽ {d['sports']}</div>
            </div>
          </div>

          <div style="border:3px solid #000;padding:.8rem 1rem;background:#f0f4f8;">
            <span style="font-size:.75rem;font-weight:800;text-transform:uppercase;color:#555;">Known For: </span>
            <span style="font-size:.9rem;font-weight:700;">{d['known_for']}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# RESULTS RENDERER
# ==============================================================================

def render_results(results: Dict[str, Any], stream: str, input_summary: str = ""):
    st.markdown('<hr style="margin:3rem 0;border-top:6px solid #000;">', unsafe_allow_html=True)

    if not results.get('eligible', True):
        st.markdown(f"""
        <div class="info-box warning">
          <div class="info-box-title">⚠ Eligibility Check Failed</div>
          <div style="font-weight:700;">{results.get('message','')}</div>
        </div>""", unsafe_allow_html=True)
        return

    st.markdown("""
    <div class="info-box advisory">
      <div class="info-box-title" style="color:#00838f;">ℹ Advisory Note</div>
      <div style="font-weight:700;">AIR (All India Rank) is a more reliable predictor than raw score or percentile. Cutoffs vary yearly.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <h2 style="font-family:'Space Grotesk',sans-serif;font-weight:900;font-size:2rem;
               text-transform:uppercase;text-align:center;margin-bottom:0;">
      🏆 Final Match Rankings
    </h2>
    <p style="text-align:center;font-weight:700;margin-bottom:1.5rem;">COMPUTED VIA HYBRID SCORING ENGINE</p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{results['overall_score']:.1f}%</div>
          <div class="metric-label">Top College Match Confidence</div>
        </div>""", unsafe_allow_html=True)

    label_map = {
        'high-chance': '✅ HIGH CHANCE',
        'moderate-borderline': '⚡ MODERATE / BORDERLINE',
        'dream-colleges': '🌟 DREAM COLLEGE'
    }

    rankings = results.get('rankings', [])
    if not rankings:
        st.info("No colleges matched. Try adjusting category, branch, or rank inputs.")
        return

    # initialise popup state
    if 'show_detail_for' not in st.session_state:
        st.session_state.show_detail_for = None

    for i, cand in enumerate(rankings):
        roi = COLLEGE_DETAILS.get(cand.college, {}).get('roi_score', 70)
        roi_color = "#06d6a0" if roi >= 90 else "#ffb703" if roi >= 75 else "#ef476f"
        roi_medal = "🥇" if roi >= 90 else "🥈" if roi >= 75 else "🥉"

        st.markdown(f"""
        <div class="college-card {cand.category}">
          <div class="rank-badge">#{i+1}</div>
          <div class="tier-label">{label_map.get(cand.category, cand.category.upper())}</div>
          <div class="college-name">{cand.college}</div>
          <div style="font-weight:700;margin-bottom:.5rem;">📍 {cand.location}</div>
          <span class="college-type-badge">{cand.type}</span>
          <span style="font-size:.8rem;font-weight:700;background:#fff;border:2px solid #000;
                       padding:.2rem .6rem;margin-right:.4rem;">ROI {roi_medal} {roi}/100</span>
          <div style="font-size:.82rem;font-weight:700;margin:.6rem 0;background:#fff;
                      border:2px solid #000;padding:.3rem .7rem;display:inline-block;">
            💡 {cand.rationale}
          </div>
          <div style="font-weight:800;font-size:1rem;text-transform:uppercase;margin-top:.6rem;">
            Program: {cand.program}
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:1rem;font-size:.85rem;font-weight:800;">
            <span>ML Probability: {cand.ml_probability:.1f}%</span>
            <span>Rules Score: {cand.rules_score:.1f}%</span>
          </div>
          <div class="custom-progress">
            <div class="custom-progress-bar" style="width:{cand.final_score}%;"></div>
          </div>
          <div style="text-align:right;font-family:'Space Grotesk',sans-serif;font-weight:900;font-size:1.2rem;">
            Final Score: {cand.final_score:.1f}%
          </div>
        </div>
        """, unsafe_allow_html=True)

        # "Know More" button per card
        col_a, col_b = st.columns([3, 1])
        with col_b:
            if st.button(f"🔍 Know More", key=f"detail_{i}_{cand.college}"):
                if st.session_state.show_detail_for == cand.college:
                    st.session_state.show_detail_for = None
                else:
                    st.session_state.show_detail_for = cand.college

        # Show popup (inline expanded detail) if this college is selected
        if st.session_state.show_detail_for == cand.college:
            render_college_popup(cand.college, stream=stream)
            if st.button("✖ Close Details", key=f"close_{i}"):
                st.session_state.show_detail_for = None
                st.rerun()

    st.markdown("""
    <div class="info-box" style="text-align:center;margin-top:2rem;font-weight:800;
         font-family:'Space Grotesk',sans-serif;text-transform:uppercase;font-size:.85rem;">
      ⚠ Predictions based on previous years' trends. Actual results may vary during final counseling.
    </div>""", unsafe_allow_html=True)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    st.set_page_config(
        page_title="North India College Predictor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # session defaults
    for key, val in [('stream_selected', None), ('results', None),
                     ('dashboard_history', []), ('show_detail_for', None),
                     ('last_inputs', {})]:
        if key not in st.session_state:
            st.session_state[key] = val

    inject_css()

    # ── NAV BAR ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="nav-bar">
      <span style="font-family:'Space Grotesk',sans-serif;font-weight:900;font-size:1.1rem;">
        🎓 College Predictor
      </span>
    </div>
    """, unsafe_allow_html=True)

    nav_cols = st.columns([1, 1, 1, 1, 1])
    pages = ["🏠 Home", "📊 Dashboard", "🔎 Predictor", "⚡ Features", "👤 About"]
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"

    for idx, (col, page) in enumerate(zip(nav_cols, pages)):
        with col:
            if st.button(page, key=f"nav_{idx}", use_container_width=True):
                st.session_state.current_page = page
                st.rerun()

    page = st.session_state.current_page

    # ── HEADER ───────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 class="main-title">North India College Predictor</h1>
    <div style="text-align:center;margin-bottom:.8rem;">
      <span class="subtitle-pill">Powered by Hybrid ML + Rules Engine</span>
    </div>
    <div style="text-align:center;margin-bottom:1.5rem;">
      <span class="region-badge">📍 Delhi · Punjab · Haryana · UP · Uttarakhand · Rajasthan · HP · J&K</span>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: HOME
    # ══════════════════════════════════════════════════════════════════════════
    if page == "🏠 Home":
        st.markdown("""
        <h2 style="font-family:'Space Grotesk',sans-serif;font-weight:900;font-size:1.8rem;text-transform:uppercase;">
          Choose Your Stream
        </h2>
        <p style="font-weight:600;font-size:1rem;margin-bottom:1.5rem;color:#444;">
          Select your stream below to begin your personalised college prediction.
          Humanities and Commerce support coming very soon!
        </p>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        streams = [
            ("Engineering", "⚙️", "IIT Delhi, NIT Kurukshetra, DTU, NSUT, Thapar", False, "🔵"),
            ("Medical",     "🩺", "AIIMS Delhi, MAMC, VMMC, KGMU, AIIMS Rishikesh", False, "🟢"),
            ("Humanities",  "📚", "Lady Shri Ram, Miranda House, Hansraj, St. Stephen's", True,  "🟡"),
            ("Commerce",    "📈", "SRCC, LSR Commerce, Hindu College, Ramjas", True,  "🟡"),
        ]
        cols = [c1, c2, c3, c4]
        for col, (name, icon, desc, coming, _) in zip(cols, streams):
            with col:
                cs_badge = '<span class="cs-badge">Coming Soon</span>' if coming else ''
                card_cls  = "stream-card coming-soon" if coming else "stream-card"
                st.markdown(f"""
                <div class="{card_cls}">
                  {cs_badge}
                  <div class="stream-icon">{icon}</div>
                  <div class="stream-title">{name}</div>
                  <div class="stream-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
                if not coming:
                    if st.button(f"Select {name}", key=f"home_{name}", use_container_width=True):
                        st.session_state.stream_selected = name
                        st.session_state.current_page = "🔎 Predictor"
                        st.rerun()
                else:
                    st.button(f"🔒 Coming Soon", key=f"home_{name}", use_container_width=True, disabled=True)

        # Quick stats row
        st.markdown("<hr style='margin:2rem 0;border-top:4px solid #000;'>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style="font-family:'Space Grotesk',sans-serif;font-weight:900;text-transform:uppercase;">
          Why Use This Tool?
        </h3>""", unsafe_allow_html=True)

        q1, q2, q3, q4 = st.columns(4)
        stats = [
            ("27+", "Colleges Modelled"),
            ("4", "Institution Tiers"),
            ("8", "States Covered"),
            ("100%", "Free to Use"),
        ]
        for col, (num, label) in zip([q1, q2, q3, q4], stats):
            with col:
                st.markdown(f"""
                <div style="border:4px solid #000;box-shadow:6px 6px 0 #000;padding:1.2rem;
                            text-align:center;background:#fff;margin:.5rem 0;">
                  <div style="font-family:'Space Grotesk',sans-serif;font-size:2.2rem;font-weight:900;">{num}</div>
                  <div style="font-size:.85rem;font-weight:800;text-transform:uppercase;color:#555;">{label}</div>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: PREDICTOR
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🔎 Predictor":
        engineering_model, category_encoder, error = load_ml_models()
        if error:
            st.toast(f"ℹ {error}", icon="ℹ️")

        # stream selector
        if st.session_state.stream_selected is None:
            st.markdown("""
            <h2 style="font-family:'Space Grotesk',sans-serif;font-weight:900;text-transform:uppercase;">
              Select Stream to Predict
            </h2>""", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            for col, (name, icon, desc) in zip([c1, c2], [
                ("Engineering", "⚙️", "IIT, NIT, IIIT, DTU, Thapar and more"),
                ("Medical",     "🩺", "AIIMS, MAMC, KGMU, GMCs and more"),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="stream-card">
                      <div class="stream-icon">{icon}</div>
                      <div class="stream-title">{name}</div>
                      <div class="stream-desc">{desc}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button(f"Select {name}", key=f"pred_{name}", use_container_width=True):
                        st.session_state.stream_selected = name
                        st.rerun()
            return

        stream = st.session_state.stream_selected
        hcol1, hcol2 = st.columns([4, 1])
        with hcol1:
            st.markdown(f"""
            <h2 style="font-family:'Space Grotesk',sans-serif;font-weight:900;text-transform:uppercase;">
              Stream: {stream}
            </h2>""", unsafe_allow_html=True)
        with hcol2:
            if st.button("↩ Start Over", use_container_width=True):
                st.session_state.stream_selected = None
                st.session_state.results = None
                st.session_state.show_detail_for = None
                st.rerun()

        st.markdown("<hr style='margin:1.5rem 0;border-top:5px solid #000;'>", unsafe_allow_html=True)

        # ── ENGINEERING FORM ────────────────────────────────────────────────
        if stream == 'Engineering':
            with st.form("eng_form"):
                st.markdown('<div class="form-section-title">Academic Details</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    jee_main  = st.number_input('JEE Main Percentile', 0.0, 100.0, 95.0, 0.1)
                    class_12  = st.number_input('Class 12 Percentage', 0.0, 100.0, 90.0, 0.1)
                    category  = st.selectbox('Category', CATEGORIES)
                with c2:
                    jee_adv   = st.number_input('JEE Advanced Rank (0 if N/A)', 0, 999999, 0)
                    state     = st.selectbox('Home State', NORTH_STATES)
                    branch    = st.selectbox('Preferred Branch', ['Computer Science', 'Electronics', 'Electrical', 'Mechanical'])

                if st.form_submit_button('🚀 Compute Hybrid Prediction', use_container_width=True):
                    data      = EngineeringInput(jee_main, jee_adv, class_12, category, state, branch)
                    features  = DataPreprocessor.preprocess_engineering(data, category_encoder)
                    ml_res    = MLPredictor.predict_engineering(features, engineering_model)
                    rules_res = RulesEngine.evaluate_engineering(data)
                    results   = RankingEngine.format_engineering_results(ml_res, rules_res)
                    st.session_state.results = results
                    st.session_state.show_detail_for = None
                    inputs = {
                        'JEE Main Percentile': jee_main, 'JEE Adv Rank': jee_adv,
                        'Class 12': class_12, 'Category': category,
                        'State': state, 'Branch': branch
                    }
                    st.session_state.last_inputs = inputs
                    if results.get('eligible', True):
                        save_to_dashboard(results, 'Engineering', inputs)
                    st.rerun()

        # ── MEDICAL FORM ─────────────────────────────────────────────────────
        elif stream == 'Medical':
            with st.form("med_form"):
                st.markdown('<div class="form-section-title">Academic Details</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    neet_score = st.number_input('NEET Score (out of 720)', 0, 720, 650, 1)
                    class_12   = st.number_input('Class 12 Percentage', 0.0, 100.0, 90.0, 0.1)
                    category   = st.selectbox('Category', CATEGORIES)
                with c2:
                    neet_rank  = st.number_input('NEET All India Rank', 0, 999999, 30000, 1000)
                    domicile   = st.selectbox('Domicile State', NORTH_STATES)
                    course     = st.selectbox('Preferred Course', ['MBBS', 'BDS'])

                if st.form_submit_button('🚀 Compute Hybrid Prediction', use_container_width=True):
                    data      = MedicalInput(neet_score, neet_rank, class_12, category, domicile, course)
                    rules_res = RulesEngine.evaluate_medical(data)
                    results   = RankingEngine.format_medical_results(rules_res)
                    st.session_state.results = results
                    st.session_state.show_detail_for = None
                    inputs = {
                        'NEET Score': neet_score, 'NEET Rank': neet_rank,
                        'Class 12': class_12, 'Category': category,
                        'Domicile': domicile, 'Course': course
                    }
                    st.session_state.last_inputs = inputs
                    if results.get('eligible', True):
                        save_to_dashboard(results, 'Medical', inputs)
                    st.rerun()

        if st.session_state.results is not None:
            render_results(st.session_state.results, stream)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "📊 Dashboard":
        st.markdown("""
        <h2 style="font-family:'Space Grotesk',sans-serif;font-weight:900;text-transform:uppercase;">
          Your Prediction Dashboard
        </h2>
        <p style="font-weight:600;color:#555;margin-bottom:1.5rem;">
          Every prediction you run is auto-saved here. Compare colleges, track your options.
        </p>""", unsafe_allow_html=True)

        history = st.session_state.get('dashboard_history', [])

        if not history:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:3rem;">
              <div style="font-size:3rem;margin-bottom:1rem;">📭</div>
              <div class="info-box-title" style="justify-content:center;">No Predictions Yet</div>
              <div style="font-weight:600;color:#555;">Run a prediction in the Predictor tab and it'll appear here automatically.</div>
            </div>""", unsafe_allow_html=True)
        else:
            # Summary cards
            total_preds  = len(history)
            best_score   = max(h['overall_score'] for h in history)
            streams_used = list(set(h['stream'] for h in history))

            m1, m2, m3 = st.columns(3)
            for col, (val, label, bg) in zip([m1, m2, m3], [
                (total_preds, "Predictions Run", "#b9fbc0"),
                (f"{best_score:.1f}%", "Best Match Score", "#ffb703"),
                (" / ".join(streams_used), "Streams Explored", "#cce3de"),
            ]):
                with col:
                    st.markdown(f"""
                    <div style="border:4px solid #000;box-shadow:6px 6px 0 #000;padding:1.5rem;
                                text-align:center;background:{bg};margin-bottom:1rem;">
                      <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:900;">{val}</div>
                      <div style="font-size:.8rem;font-weight:800;text-transform:uppercase;color:#444;">{label}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<hr style='margin:1.5rem 0;border-top:4px solid #000;'>", unsafe_allow_html=True)

            # History entries
            for idx, entry in enumerate(history):
                with st.expander(
                    f"#{idx+1} · {entry['stream']} · {entry['timestamp']} · Top: {entry['top_college']} ({entry['overall_score']:.1f}%)",
                    expanded=(idx == 0)
                ):
                    # Input summary
                    st.markdown("""
                    <div style="font-family:'Space Grotesk',sans-serif;font-weight:900;
                                font-size:1rem;text-transform:uppercase;margin-bottom:.8rem;">
                      📋 Inputs Used
                    </div>""", unsafe_allow_html=True)
                    inp_cols = st.columns(len(entry['inputs']))
                    for col, (k, v) in zip(inp_cols, entry['inputs'].items()):
                        with col:
                            st.markdown(f"""
                            <div style="border:2px solid #000;padding:.4rem .6rem;background:#f9f9f9;margin-bottom:.4rem;">
                              <div style="font-size:.65rem;font-weight:800;text-transform:uppercase;color:#666;">{k}</div>
                              <div style="font-weight:800;font-size:.95rem;">{v}</div>
                            </div>""", unsafe_allow_html=True)

                    # Top 5 colleges table
                    st.markdown("""
                    <div style="font-family:'Space Grotesk',sans-serif;font-weight:900;
                                font-size:1rem;text-transform:uppercase;margin:.8rem 0 .5rem;">
                      🏆 Top Matched Colleges
                    </div>""", unsafe_allow_html=True)

                    rankings = entry.get('rankings', [])[:5]
                    if rankings:
                        rows = []
                        for r in rankings:
                            roi = COLLEGE_DETAILS.get(r.college, {}).get('roi_score', 70)
                            medal = "🥇" if roi >= 90 else "🥈" if roi >= 75 else "🥉"
                            rows.append({
                                'College': r.college,
                                'Location': r.location,
                                'Category': r.category.replace('-', ' ').title(),
                                'Final Score': f"{r.final_score:.1f}%",
                                'ROI': f"{medal} {roi}/100"
                            })
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    # Compare top 3 ROI scores as bar chart
                    if len(rankings) >= 2:
                        st.markdown("""
                        <div style="font-family:'Space Grotesk',sans-serif;font-weight:900;
                                    font-size:.9rem;text-transform:uppercase;margin:.8rem 0 .3rem;">
                          📊 ROI Comparison (Top Colleges)
                        </div>""", unsafe_allow_html=True)
                        chart_data = {
                            r.college: COLLEGE_DETAILS.get(r.college, {}).get('roi_score', 70)
                            for r in rankings[:5]
                        }
                        df_chart = pd.DataFrame.from_dict(
                            chart_data, orient='index', columns=['ROI Score']
                        )
                        st.bar_chart(df_chart, use_container_width=True, height=220)

            # Clear history
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑 Clear All Dashboard History", use_container_width=False):
                st.session_state.dashboard_history = []
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: FEATURES
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "⚡ Features":
        st.markdown("""
        <h2 style="font-family:'Space Grotesk',sans-serif;font-weight:900;text-transform:uppercase;">
          What Makes This Tool Unique
        </h2>""", unsafe_allow_html=True)

        # How to use — step by step
        st.markdown("""
        <div style="border:4px solid #000;box-shadow:8px 8px 0 #000;padding:1.5rem;
                    background:#ffd166;margin-bottom:2rem;">
          <div style="font-family:'Space Grotesk',sans-serif;font-weight:900;font-size:1.3rem;
                      text-transform:uppercase;margin-bottom:1rem;">
            🗺 How To Use — 3 Simple Steps
          </div>
        </div>""", unsafe_allow_html=True)

        steps = [
            ("1️⃣", "Choose Your Stream", "Pick Engineering or Medical on the Home page. Humanities & Commerce coming very soon!"),
            ("2️⃣", "Enter Your Scores", "Fill in your JEE/NEET rank, percentile, category, home state and preferred branch."),
            ("3️⃣", "Get Ranked Results", "The Hybrid Engine predicts your best-fit colleges ranked by match confidence + ROI score."),
        ]
        s1, s2, s3 = st.columns(3)
        for col, (icon, title, body) in zip([s1, s2, s3], steps):
            with col:
                st.markdown(f"""
                <div style="border:4px solid #000;box-shadow:6px 6px 0 #000;padding:1.5rem;
                            background:#fff;text-align:center;min-height:200px;">
                  <div style="font-size:2.5rem;margin-bottom:.5rem;">{icon}</div>
                  <div style="font-family:'Space Grotesk',sans-serif;font-weight:900;
                              font-size:1.1rem;text-transform:uppercase;margin-bottom:.6rem;">{title}</div>
                  <div style="font-size:.9rem;font-weight:600;color:#444;">{body}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature highlights
        st.markdown("""
        <h3 style="font-family:'Space Grotesk',sans-serif;font-weight:900;text-transform:uppercase;">
          ⚡ Key Features
        </h3>""", unsafe_allow_html=True)

        features = [
            ("🤖", "Hybrid Scoring Engine",
             "Final Score = (0.6 × ML Probability) + (0.4 × Rules Score). ML and rule-based systems work together — neither alone is enough."),
            ("🗺", "Home State Quota Intelligence",
             "Every college's HS vs OS quota is modelled individually. DTU Delhi applies 3.5× relaxation for Delhi students — we know this."),
            ("🧮", "Percentile → AIR Conversion",
             "Enter your JEE percentile; we auto-convert to estimated AIR using the formula: AIR = (1 − percentile/100) × 14,00,000."),
            ("🏛", "4-Tier College Classification",
             "[GOVT-IIT] uses JEE Advanced only. [GOVT-NIT/IIIT] uses JEE Mains with 50% HS seats. [STATE] has up to 85% HS quota. [PVT] varies."),
            ("💰", "ROI Ranking",
             "Each college gets an ROI score based on avg package, placement %, and annual fees. See which college gives you the best return on investment."),
            ("🔍", "College Detail Popup",
             "Click 'Know More' on any result to see NIRF rank, NAAC grade, hostel info, top recruiters, international collaborations, and more."),
            ("📊", "Auto-Save Dashboard",
             "Every prediction is saved to your personal dashboard automatically. Compare results across sessions and track your shortlist."),
            ("🎓", "Medical + Engineering",
             "Full support for both streams with separate logic engines. NEET rank → medical colleges. JEE Mains/Advanced → engineering colleges."),
        ]

        for i in range(0, len(features), 2):
            c1, c2 = st.columns(2)
            for col, (icon, title, body) in zip([c1, c2], features[i:i+2]):
                with col:
                    st.markdown(f"""
                    <div style="border:4px solid #000;box-shadow:5px 5px 0 #000;padding:1.2rem;
                                background:#fff;margin-bottom:1rem;min-height:130px;">
                      <div style="font-family:'Space Grotesk',sans-serif;font-weight:900;
                                  font-size:1rem;text-transform:uppercase;margin-bottom:.4rem;">
                        {icon} {title}
                      </div>
                      <div style="font-size:.88rem;font-weight:600;color:#333;line-height:1.55;">{body}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tech stack
        st.markdown("""
        <h3 style="font-family:'Space Grotesk',sans-serif;font-weight:900;text-transform:uppercase;">
          🛠 Tech Stack
        </h3>""", unsafe_allow_html=True)

        techs = [("Python 3.11", "Core language"), ("Streamlit", "UI framework"),
                 ("Scikit-learn", "ML model"), ("Joblib", "Model serialisation"),
                 ("NumPy / Pandas", "Data processing"), ("CSS / Space Grotesk", "Neo-Brutalist UI")]
        t_cols = st.columns(3)
        for i, (tech, role) in enumerate(techs):
            with t_cols[i % 3]:
                st.markdown(f"""
                <div style="border:3px solid #000;box-shadow:4px 4px 0 #000;padding:.8rem 1rem;
                            background:#f4f0ec;margin-bottom:.8rem;">
                  <div style="font-family:'Space Grotesk',sans-serif;font-weight:900;font-size:1rem;">{tech}</div>
                  <div style="font-size:.8rem;font-weight:700;color:#555;text-transform:uppercase;">{role}</div>
                </div>""", unsafe_allow_html=True)

        # FAQ
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style="font-family:'Space Grotesk',sans-serif;font-weight:900;text-transform:uppercase;">
          ❓ Frequently Asked Questions
        </h3>""", unsafe_allow_html=True)

        faqs = [
            ("Is this tool 100% accurate?",
             "No prediction tool is 100% accurate. This uses historical JoSAA/MCC cutoff data + ML patterns. Always cross-check with official JoSAA/MCC portals before making final decisions."),
            ("What if I don't have a JEE Advanced rank?",
             "Leave it as 0. The system automatically uses your JEE Main percentile to estimate your All India Rank (AIR) and routes you to NIT/IIIT/State colleges instead of IITs."),
            ("Why does my home state matter?",
             "NITs reserve 50% seats for home state candidates. Delhi state colleges (DTU, NSUT) reserve 85% for Delhi students. This massively changes your cutoff threshold."),
            ("What does ROI score mean?",
             "ROI (Return on Investment) score rates each college based on avg placement package vs annual fee. A score of 90+ means your salary can recover the full tuition cost in under 3 months."),
            ("When will Humanities & Commerce be added?",
             "Both streams are in active development! CUET-based prediction for DU colleges (LSR, SRCC, Miranda House, etc.) will be released very soon."),
        ]
        for q, a in faqs:
            with st.expander(f"❓ {q}"):
                st.markdown(f"<div style='font-weight:600;font-size:.95rem;color:#333;'>{a}</div>",
                            unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: ABOUT
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "👤 About":
        st.markdown("""
        <h2 style="font-family:'Space Grotesk',sans-serif;font-weight:900;text-transform:uppercase;">
          About This Project
        </h2>""", unsafe_allow_html=True)

        # Developer bio card
        st.markdown("""
        <div style="border:4px solid #000;box-shadow:10px 10px 0 #000;padding:2.5rem;
                    background:#fff;margin-bottom:2rem;display:flex;gap:2rem;align-items:flex-start;">
          <div style="flex:1;">
            <div style="background:#000;color:#fff;display:inline-block;padding:.3rem .8rem;
                        font-family:'Space Grotesk',sans-serif;font-weight:900;font-size:.8rem;
                        text-transform:uppercase;margin-bottom:.8rem;">
              Builder
            </div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:900;margin-bottom:.3rem;">
              Shivansh Upadhyay
            </div>
            <div style="font-size:1rem;font-weight:700;color:#555;margin-bottom:1rem;">
              📍 North India  ·  Student & Developer  ·  EdTech Enthusiast
            </div>
            <div style="font-size:.95rem;font-weight:600;color:#333;line-height:1.7;margin-bottom:1.2rem;">
              I built this tool because I went through JEE counseling myself and felt completely lost —
              percentile vs rank confusion, home state quota calculations, IIT vs NIT routing...
              no single tool handled all of it properly for North India students.
              <br><br>
              So I built one. One that combines a real Machine Learning model with a strict
              Business Rules Engine that actually understands how JoSAA, CSAB, and state
              counseling quotas work. It's free, open, and built for students just like me.
            </div>
            <div style="display:flex;gap:.8rem;flex-wrap:wrap;">
              <span style="border:3px solid #000;padding:.3rem .8rem;font-weight:800;
                           font-size:.85rem;background:#b9fbc0;">🐍 Python Developer</span>
              <span style="border:3px solid #000;padding:.3rem .8rem;font-weight:800;
                           font-size:.85rem;background:#cce3de;">🤖 ML Enthusiast</span>
              <span style="border:3px solid #000;padding:.3rem .8rem;font-weight:800;
                           font-size:.85rem;background:#ffd166;">🎓 EdTech Builder</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # About the tool
        st.markdown("""
        <h3 style="font-family:'Space Grotesk',sans-serif;font-weight:900;text-transform:uppercase;">
          About The Tool
        </h3>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="border:4px solid #000;box-shadow:6px 6px 0 #000;padding:1.8rem;
                    background:#f4f0ec;margin-bottom:1.5rem;">
          <p style="font-size:.95rem;font-weight:600;color:#333;line-height:1.8;margin:0;">
            The <strong>North India College Admission Predictor</strong> is a hybrid AI + rules-based
            platform focused specifically on the 8-state North India educational cluster: Delhi, Punjab,
            Haryana, Uttar Pradesh, Uttarakhand, Rajasthan, Himachal Pradesh, and Jammu & Kashmir.
            <br><br>
            It models 27+ institutions across 4 tiers — IITs (JEE Advanced routing),
            NITs/IIITs (JEE Mains with 50% home-state quota), State Government colleges
            (up to 85% home-state bias), and Top Private universities. Each college has
            individual quota multipliers, category modifiers (General 1.0× → ST 4.5×),
            and branch-specific adjustments.
            <br><br>
            The Hybrid Scoring Engine combines:
            <strong>Final Score = (0.6 × ML Probability) + (0.4 × Rules Score)</strong>
            — giving you confidence that both historical pattern recognition and
            real admission logic back your result.
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Mission & disclaimer
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div style="border:4px solid #000;box-shadow:6px 6px 0 #000;padding:1.5rem;background:#b9fbc0;">
              <div style="font-family:'Space Grotesk',sans-serif;font-weight:900;font-size:1.1rem;
                          text-transform:uppercase;margin-bottom:.6rem;">🎯 Mission</div>
              <div style="font-size:.9rem;font-weight:600;color:#333;line-height:1.7;">
                Reduce counseling anxiety for North India's 15+ lakh JEE/NEET aspirants
                by giving them data-driven, quota-aware, ROI-conscious college recommendations.
                Free. Always.
              </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div style="border:4px solid #000;box-shadow:6px 6px 0 #000;padding:1.5rem;background:#ffcfd2;">
              <div style="font-family:'Space Grotesk',sans-serif;font-weight:900;font-size:1.1rem;
                          text-transform:uppercase;margin-bottom:.6rem;">⚠ Disclaimer</div>
              <div style="font-size:.9rem;font-weight:600;color:#333;line-height:1.7;">
                Predictions are based on historical trends and should not replace
                official JoSAA / MCC / state counseling portals.
                Always verify cutoffs on official sources before making decisions.
              </div>
            </div>""", unsafe_allow_html=True)

        # Roadmap
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style="font-family:'Space Grotesk',sans-serif;font-weight:900;text-transform:uppercase;">
          🗺 Roadmap
        </h3>""", unsafe_allow_html=True)

        roadmap = [
            ("✅ Done", "Engineering stream (JEE Mains + Advanced)", "#b9fbc0"),
            ("✅ Done", "Medical stream (NEET MBBS/BDS)", "#b9fbc0"),
            ("✅ Done", "College detail popup with ROI, placements, hostel info", "#b9fbc0"),
            ("✅ Done", "Dashboard with auto-save + comparison", "#b9fbc0"),
            ("🔄 In Progress", "Humanities stream (CUET + DU colleges)", "#fbf8cc"),
            ("🔄 In Progress", "Commerce stream (SRCC, LSR, CUET-based)", "#fbf8cc"),
            ("📋 Planned", "Live JoSAA cutoff data integration", "#f4f0ec"),
            ("📋 Planned", "Round-wise counseling simulator", "#f4f0ec"),
            ("📋 Planned", "Mobile-friendly PWA version", "#f4f0ec"),
        ]
        for status, item, bg in roadmap:
            st.markdown(f"""
            <div style="border:3px solid #000;box-shadow:3px 3px 0 #000;padding:.7rem 1rem;
                        background:{bg};margin-bottom:.5rem;display:flex;gap:1rem;align-items:center;">
              <span style="font-family:'Space Grotesk',sans-serif;font-weight:900;
                           font-size:.8rem;min-width:110px;text-transform:uppercase;">{status}</span>
              <span style="font-size:.9rem;font-weight:600;">{item}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;margin-top:2.5rem;padding:1.5rem;
                    border:4px solid #000;box-shadow:6px 6px 0 #000;
                    background:#000;color:#fff;
                    font-family:'Space Grotesk',sans-serif;font-weight:900;
                    font-size:1rem;text-transform:uppercase;">
          Built with ❤ for North India's Students · 100% Free · Open Source Spirit
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()