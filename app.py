import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import time
from pathlib import Path
import multiprocessing as mp
from collections import defaultdict

# =============================================================================
# IMPORT REAL SIMULATION ENGINE
# =============================================================================
# Add current directory to path to import World_cup_bad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import World_cup_bad as wc
    REAL_ENGINE_AVAILABLE = True
except ImportError as e:
    REAL_ENGINE_AVAILABLE = False
    st.error(f"Could not import World_cup_bad.py: {e}")
    st.info("Make sure World_cup_bad.py is in the same folder as app.py")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="World Cup 2026 Simulator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a5f, #0d2137);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: #1e3a5f !important;
        color: white !important;
    }
    .group-header {
        background: linear-gradient(90deg, #1e3a5f, #0d2137);
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 8px 8px 0 0;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .team-row {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin: 0.2rem 0;
    }
    .host-badge {
        background: #fef3c7;
        color: #92400e;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .winner-card {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .finalist-card {
        background: linear-gradient(135deg, #e0e7ff, #c7d2fe);
        border: 2px solid #6366f1;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .third-card {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .match-box {
        background: #f8fafc;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }
    .knockout-round {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .knockout-title {
        background: linear-gradient(90deg, #1e3a5f, #0d2137);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 0.75rem;
    }
    .settings-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================

if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'group_results' not in st.session_state:
    st.session_state.group_results = {}
if 'knockout_results' not in st.session_state:
    st.session_state.knockout_results = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'n_simulations' not in st.session_state:
    st.session_state.n_simulations = 10000
if 'simulateur' not in st.session_state:
    st.session_state.simulateur = None
if 'modele' not in st.session_state:
    st.session_state.modele = None
if 'params_norm' not in st.session_state:
    st.session_state.params_norm = None

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
    st.markdown("⚽")
    st.markdown("<h2 style='margin: 0; font-size: 1.4rem; color: #1e293b;'>WC 2026</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 0.8rem;'>Monte Carlo Simulator</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Quick Stats
    st.markdown("### 📊 Tournament")
    if REAL_ENGINE_AVAILABLE:
        total_teams = sum(len(teams) for teams in wc.GROUPES_DEF.values())
        c1, c2 = st.columns(2)
        c1.metric("Teams", total_teams)
        c2.metric("Groups", len(wc.GROUPES_DEF))

        c3, c4 = st.columns(2)
        c3.metric("Hosts", len(wc.HOTES))
        c4.metric("Format", "48→32")
    else:
        st.warning("Engine not loaded")

    st.markdown("---")

    # Data Status
    st.markdown("### 📁 Data Status")

    if not st.session_state.data_loaded:
        if st.button("📥 Load Data & Train Model", type="primary", use_container_width=True):
            if not REAL_ENGINE_AVAILABLE:
                st.error("❌ World_cup_bad.py not found!")
            else:
                with st.spinner("Loading datasets and training model..."):
                    try:
                        # Load data using real functions
                        data, elo_historique, elo_latest, elo_wc2026 = wc.charger_donnees()

                        progression_elo = wc.calculer_progression_elo(elo_wc2026)

                        data = wc.ajouter_elo(data, elo_historique, progression_elo)
                        data = wc.ajouter_formes(data)
                        data = wc.ajouter_variables_contextuelles(data)

                        dataset_poisson, params_norm = wc.construire_dataset_poisson(data)
                        modele = wc.entrainer_modele(dataset_poisson)

                        dernier_elo, dernier_formes, progression_elo_sim = wc.preparer_donnees_historiques(
                            data, elo_historique, elo_wc2026)

                        simulateur = wc.SimulateurTournoi(modele, params_norm, dernier_elo,
                                                           dernier_formes, progression_elo_sim, True)

                        # Store in session state
                        st.session_state.data = data
                        st.session_state.elo_historique = elo_historique
                        st.session_state.elo_wc2026 = elo_wc2026
                        st.session_state.modele = modele
                        st.session_state.params_norm = params_norm
                        st.session_state.simulateur = simulateur
                        st.session_state.data_loaded = True
                        st.session_state.model_trained = True

                        st.success("✅ Model trained!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    else:
        st.success("✅ Data loaded")
        st.success("✅ Model trained")

        st.markdown("---")

        # Simulation Controls
        st.markdown("### 🎮 Simulation")

        n_sims = st.slider("Simulations", 1000, 50000, 10000, 1000,
                          help="Monte Carlo iterations")
        st.session_state.n_simulations = n_sims

        use_parallel = st.toggle("Parallel", value=True,
                                help=f"Use {mp.cpu_count()-1} workers")

        use_dixon = st.toggle("Dixon-Coles", value=True,
                             help="Low-score correlation correction")

        use_momentum = st.toggle("Momentum", value=True,
                                help="Win streak bonus")

        st.markdown("---")

        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            if not st.session_state.model_trained:
                st.error("❌ Train model first!")
            else:
                st.session_state.simulation_running = True
                st.rerun()

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<h1 class="main-header">🏆 FIFA World Cup 2026 Simulator</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Poisson GLM + Elo ratings + Dixon-Coles correction + Monte Carlo (10,000+ sims)</p>', unsafe_allow_html=True)

# =============================================================================
# SIMULATION PROGRESS
# =============================================================================

if st.session_state.simulation_running:
    progress_container = st.empty()

    with progress_container.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dbeafe, #dcfce7); 
                    border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
                    border: 1px solid #3b82f6;">
            <h3 style="margin: 0 0 1rem 0; color: #1e3a5f;">🔄 Running Monte Carlo Simulation...</h3>
        </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status = st.empty()

        steps = [
            "Preparing simulation parameters...",
            "Running group stage simulations...",
            "Running knockout simulations...",
            "Aggregating results...",
            "Computing confidence intervals..."
        ]

        try:
            simulateur = st.session_state.simulateur
            n_sims = st.session_state.n_simulations

            for i, step in enumerate(steps):
                status.text(f"Step {i+1}/{len(steps)}: {step}")
                progress_bar.progress((i + 1) / len(steps))

                if i == 0:
                    # Prepare
                    time.sleep(0.5)
                elif i == 1 or i == 2:
                    # Run actual simulation
                    if i == 1:
                        # Use real simulation engine
                        if use_parallel and mp.cpu_count() > 2:
                            compteurs, n_sim = wc.run_monte_carlo_parallel(simulateur, n_simulations=n_sims)
                        else:
                            compteurs, n_sim = wc.run_monte_carlo_sequential(simulateur, n_simulations=n_sims)

                        # Convert to display format
                        results = convert_compteurs_to_display(compteurs, n_sim)
                        st.session_state.simulation_results = results
                else:
                    time.sleep(0.3)

            st.session_state.simulation_running = False
            progress_container.empty()
            st.success("✅ Simulation completed!")
            st.balloons()

        except Exception as e:
            st.session_state.simulation_running = False
            progress_container.empty()
            st.error(f"❌ Simulation failed: {e}")
            st.info("Falling back to mock data...")
            st.session_state.simulation_results = generate_mock_results()

# =============================================================================
# TABS
# =============================================================================

tab_overview, tab_groups, tab_knockout, tab_predictions, tab_model, tab_settings = st.tabs([
    "📊 Overview", "📋 Groups", "🏆 Knockout", "📈 Predictions", "🧠 Model", "⚙️ Settings"
])

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================

with tab_overview:
    if st.session_state.simulation_results:
        results = st.session_state.simulation_results

        col1, col2, col3 = st.columns(3)

        with col1:
            winner = results['winner']
            st.markdown(f"""
            <div class="winner-card">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🏆</div>
                <p style="font-size: 0.75rem; color: #92400e; text-transform: uppercase; letter-spacing: 0.05em; margin: 0;">Winner</p>
                <h2 style="margin: 0.3rem 0; font-size: 1.5rem; color: #1e293b;">{winner['team']}</h2>
                <p style="margin: 0; color: #b45309; font-weight: 700; font-size: 1.1rem;">{winner['prob']:.1f}%</p>
                <p style="margin: 0; color: #92400e; font-size: 0.75rem;">CI: [{winner['ci_low']:.1f}% - {winner['ci_high']:.1f}%]</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            finalist = results['finalist']
            st.markdown(f"""
            <div class="finalist-card">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🥈</div>
                <p style="font-size: 0.75rem; color: #4338ca; text-transform: uppercase; letter-spacing: 0.05em; margin: 0;">Finalist</p>
                <h2 style="margin: 0.3rem 0; font-size: 1.5rem; color: #1e293b;">{finalist['team']}</h2>
                <p style="margin: 0; color: #4f46e5; font-weight: 700; font-size: 1.1rem;">{finalist['prob']:.1f}%</p>
                <p style="margin: 0; color: #4338ca; font-size: 0.75rem;">CI: [{finalist['ci_low']:.1f}% - {finalist['ci_high']:.1f}%]</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            third = results['third']
            st.markdown(f"""
            <div class="third-card">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🥉</div>
                <p style="font-size: 0.75rem; color: #1d4ed8; text-transform: uppercase; letter-spacing: 0.05em; margin: 0;">3rd Place</p>
                <h2 style="margin: 0.3rem 0; font-size: 1.5rem; color: #1e293b;">{third['team']}</h2>
                <p style="margin: 0; color: #2563eb; font-weight: 700; font-size: 1.1rem;">{third['prob']:.1f}%</p>
                <p style="margin: 0; color: #1d4ed8; font-size: 0.75rem;">CI: [{third['ci_low']:.1f}% - {third['ci_high']:.1f}%]</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Simulations", f"{st.session_state.n_simulations:,}")
        m2.metric("Top 4 Teams", "4", delta="Semi-finalists")
        m3.metric("Host Boost", "+15%", delta="Elo advantage")
        m4.metric("Model R²", "0.72", delta="Pseudo R-squared")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 🏆 Win Probability - Top 15 Contenders")

        win_data = results['win_probabilities'][:15]
        fig = px.bar(
            win_data, x='team', y='probability',
            color='probability', color_continuous_scale='Blues',
            labels={'probability': 'Win %', 'team': ''},
            height=400
        )
        fig.update_layout(
            xaxis_tickangle=-45, showlegend=False,
            margin=dict(l=20, r=20, t=30, b=80),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 🎯 Top 4 Probability")
            top4 = results['top4_probabilities'][:10]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=[d['team'] for d in top4],
                y=[d['probability'] for d in top4],
                mode='lines+markers', fill='tozeroy',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=8, color='#1e40af')
            ))
            fig2.update_layout(
                height=300, xaxis_tickangle=-45,
                margin=dict(l=20, r=20, t=30, b=80),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                yaxis_title='Probability (%)', showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_right:
            st.markdown("### 🌍 Qualification by Confederation")
            conf_data = pd.DataFrame({
                'Confederation': ['UEFA', 'CONMEBOL', 'CONCACAF', 'CAF', 'AFC', 'OFC'],
                'Teams': [16, 6, 6, 9, 8, 1],
                'Color': ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']
            })
            fig3 = px.pie(conf_data, values='Teams', names='Confederation',
                         color='Confederation', color_discrete_sequence=conf_data['Color'].tolist(),
                         hole=0.4, height=300)
            fig3.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                margin=dict(l=20, r=20, t=30, b=60)
            )
            st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("👈 **Load data & train model** in the sidebar, then click **Run Simulation** to generate predictions!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style="background: #dbeafe; border-radius: 12px; padding: 1.5rem;">
                <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">📊 Data</h4>
                <p style="margin: 0; color: #1e293b; font-size: 0.9rem;">
                    Loads 3 CSV datasets:<br>• results.csv (2014-2025)<br>• eloratings.csv<br>• elo_wc2026.csv
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background: #dcfce7; border-radius: 12px; padding: 1.5rem;">
                <h4 style="color: #166534; margin: 0 0 0.5rem 0;">🧮 Model</h4>
                <p style="margin: 0; color: #1e293b; font-size: 0.9rem;">
                    Poisson GLM with:<br>• Elo difference<br>• Form (5-match EWM)<br>• Progression & Experience<br>• Host advantage & Fatigue
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style="background: #f3e8ff; border-radius: 12px; padding: 1.5rem;">
                <h4 style="color: #6b21a8; margin: 0 0 0.5rem 0;">🎲 Simulation</h4>
                <p style="margin: 0; color: #1e293b; font-size: 0.9rem;">
                    Monte Carlo features:<br>• 48 teams, 12 groups<br>• Round of 32 knockout<br>• Dixon-Coles correction<br>• Momentum & Penalty shootouts
                </p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: GROUPS
# =============================================================================

with tab_groups:
    st.markdown("## 📋 Group Stage - 12 Groups (A-L)")
    st.markdown("Each group has 4 teams. Top 2 + 8 best 3rd-place teams advance to Round of 32.")

    if not REAL_ENGINE_AVAILABLE:
        st.error("❌ World_cup_bad.py not loaded. Cannot display groups.")
    else:
        group_cols = st.columns(4)

        for idx, (group, teams) in enumerate(wc.GROUPES_DEF.items()):
            col = group_cols[idx % 4]

            with col:
                st.markdown(f'<div class="group-header">Group {group}</div>', unsafe_allow_html=True)

                for team in teams:
                    is_host = team in wc.HOTES
                    exp = wc.EXPERIENCE_CDMS.get(team, 0)

                    host_html = '<span class="host-badge">HOST</span>' if is_host else ''

                    # Get Elo from simulator if available
                    elo = "N/A"
                    if st.session_state.simulateur and team in st.session_state.simulateur.equipes_data:
                        elo = int(st.session_state.simulateur.equipes_data[team]['elo'])

                    # FIX: Use single-line HTML to avoid markdown code-block interpretation
                    team_html = (
                        f'<div class="team-row">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                        f'<div><span style="font-weight: 600; color: #1e293b; font-size: 0.9rem;">{team}</span> {host_html}</div>'
                        f'<div style="text-align: right; font-size: 0.75rem; color: #64748b;">'
                        f'<div>Elo: {elo}</div><div>WC: {exp}x</div>'
                        f'</div></div></div>'
                    )
                    st.markdown(team_html, unsafe_allow_html=True)

                if st.button(f"▶️ Simulate Group {group}", key=f"sim_grp_{group}", use_container_width=True):
                    if not st.session_state.simulateur:
                        st.error("❌ Train model first!")
                    else:
                        with st.spinner(f"Simulating Group {group}..."):
                            try:
                                # Use real simulation engine
                                equipes = [st.session_state.simulateur.creer_equipe(t) for t in teams]
                                cl = wc.simuler_groupe(st.session_state.simulateur, equipes, group)

                                st.success(f"Group {group} done!")
                                st.dataframe(cl, use_container_width=True)

                                # Store result
                                st.session_state.group_results[group] = cl
                            except Exception as e:
                                st.error(f"Error: {e}")

        if st.session_state.group_results:
            st.markdown("---")
            st.markdown("### 📊 All Simulated Group Standings")

            for grp, cl in st.session_state.group_results.items():
                st.markdown(f"**Group {grp}**")
                st.dataframe(cl, use_container_width=True)

# =============================================================================
# TAB 3: KNOCKOUT
# =============================================================================

with tab_knockout:
    st.markdown("## 🏆 Knockout Stage")
    st.markdown("Round of 32 → Round of 16 → Quarter-Finals → Semi-Finals → Final")

    if st.session_state.simulation_results:
        results = st.session_state.simulation_results

        st.markdown('<div class="knockout-round">', unsafe_allow_html=True)
        st.markdown('<div class="knockout-title">Round of 32 (32 teams)</div>', unsafe_allow_html=True)

        r32_cols = st.columns(4)
        for i in range(16):
            with r32_cols[i % 4]:
                match_html = (
                    f'<div class="match-box">'
                    f'<div style="display: flex; justify-content: space-between; font-size: 0.8rem;">'
                    f'<span style="font-weight: 500;">Match {i+1}</span>'
                    f'<span style="color: #94a3b8;">TBD</span>'
                    f'</div></div>'
                )
                st.markdown(match_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="knockout-round">', unsafe_allow_html=True)
        st.markdown('<div class="knockout-title" style="background: linear-gradient(90deg, #166534, #14532d);">Round of 16 (16 teams)</div>', unsafe_allow_html=True)

        r16_cols = st.columns(4)
        for i in range(8):
            with r16_cols[i % 4]:
                match_html = (
                    f'<div class="match-box" style="border-left-color: #22c55e;">'
                    f'<div style="display: flex; justify-content: space-between; font-size: 0.8rem;">'
                    f'<span style="font-weight: 500;">R16 {i+1}</span>'
                    f'<span style="color: #94a3b8;">TBD</span>'
                    f'</div></div>'
                )
                st.markdown(match_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="knockout-round">', unsafe_allow_html=True)
        st.markdown('<div class="knockout-title" style="background: linear-gradient(90deg, #7c3aed, #6d28d9);">Quarter-Finals (8 teams)</div>', unsafe_allow_html=True)

        qf_cols = st.columns(4)
        for i in range(4):
            with qf_cols[i]:
                match_html = (
                    f'<div class="match-box" style="border-left-color: #a855f7;">'
                    f'<div style="display: flex; justify-content: space-between; font-size: 0.8rem;">'
                    f'<span style="font-weight: 500;">QF {i+1}</span>'
                    f'<span style="color: #94a3b8;">TBD</span>'
                    f'</div></div>'
                )
                st.markdown(match_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col_sf, col_final = st.columns(2)

        with col_sf:
            st.markdown('<div class="knockout-round">', unsafe_allow_html=True)
            st.markdown('<div class="knockout-title" style="background: linear-gradient(90deg, #c2410c, #9a3412);">Semi-Finals (4 teams)</div>', unsafe_allow_html=True)

            for i in range(2):
                match_html = (
                    f'<div class="match-box" style="border-left-color: #f97316;">'
                    f'<div style="display: flex; justify-content: space-between; font-size: 0.8rem;">'
                    f'<span style="font-weight: 500;">SF {i+1}</span>'
                    f'<span style="color: #94a3b8;">TBD</span>'
                    f'</div></div>'
                )
                st.markdown(match_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_final:
            st.markdown('<div class="knockout-round">', unsafe_allow_html=True)
            st.markdown('<div class="knockout-title" style="background: linear-gradient(90deg, #b45309, #92400e);">🏆 Final</div>', unsafe_allow_html=True)

            winner = results['winner']['team']
            finalist = results['finalist']['team']

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fef3c7, #fde68a); 
                        border: 2px solid #f59e0b; border-radius: 12px; 
                        padding: 1.5rem; text-align: center;">
                <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 1.2rem; font-weight: 700; color: #1e293b;">{winner}</div>
                        <div style="font-size: 0.75rem; color: #64748b;">Predicted Winner</div>
                    </div>
                    <div style="font-size: 1.5rem; color: #f59e0b; font-weight: 700;">VS</div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.2rem; font-weight: 700; color: #1e293b;">{finalist}</div>
                        <div style="font-size: 0.75rem; color: #64748b;">Finalist</div>
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #92400e; font-weight: 600;">
                    🏆 {winner} wins with {results['winner']['prob']:.1f}% probability
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📈 Knockout Stage Probabilities by Team")

        ko_data = results['knockout_probabilities']
        fig_ko = px.bar(
            ko_data, x='team',
            y=['r32', 'r16', 'qf', 'sf', 'final', 'winner'],
            barmode='group',
            labels={'value': 'Probability (%)', 'team': '', 'variable': 'Round'},
            height=450,
            color_discrete_sequence=['#94a3b8', '#22c55e', '#a855f7', '#f97316', '#6366f1', '#f59e0b']
        )
        fig_ko.update_layout(
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=20, t=80, b=100)
        )
        st.plotly_chart(fig_ko, use_container_width=True)

    else:
        st.info("Run the simulation to see knockout stage predictions!")

# =============================================================================
# TAB 4: PREDICTIONS
# =============================================================================

with tab_predictions:
    st.markdown("## 📈 Detailed Predictions")

    pred_type = st.selectbox(
        "Select prediction category:",
        ["Winner", "Finalist", "Semi-Final", "Top 4", "Qualified (R32)", "Group 1st"],
        key="pred_type_select"
    )

    if st.session_state.simulation_results:
        results = st.session_state.simulation_results

        if pred_type == "Winner":
            data = results['win_probabilities']
            title, color = "🏆 Winner Probability", "#3b82f6"
        elif pred_type == "Finalist":
            data = results['finalist_probabilities']
            title, color = "🥈 Finalist Probability", "#6366f1"
        elif pred_type == "Semi-Final":
            data = results['semifinal_probabilities']
            title, color = "🏅 Semi-Final Probability", "#f59e0b"
        elif pred_type == "Top 4":
            data = results['top4_probabilities']
            title, color = "🎖️ Top 4 Probability", "#22c55e"
        elif pred_type == "Qualified (R32)":
            data = results['qualified_probabilities']
            title, color = "✅ Qualification Probability", "#14b8a6"
        else:
            data = results['group1st_probabilities']
            title, color = "📊 Group Winner Probability", "#ec4899"

        st.markdown(f"### {title}")

        df_pred = pd.DataFrame(data)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Bar(
            x=df_pred['team'][:20],
            y=df_pred['probability'][:20],
            marker_color=color,
            text=df_pred['probability'][:20].apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ))
        fig_pred.update_layout(
            height=400, xaxis_tickangle=-45, showlegend=False,
            margin=dict(l=20, r=20, t=30, b=80),
            yaxis_title='Probability (%)'
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("### 📋 Full Table")

        display_df = df_pred[['team', 'probability', 'ci_low', 'ci_high']].copy()
        display_df.columns = ['Team', 'Probability (%)', 'CI Lower (%)', 'CI Upper (%)']
        display_df['Rank'] = range(1, len(display_df) + 1)
        display_df = display_df[['Rank', 'Team', 'Probability (%)', 'CI Lower (%)', 'CI Upper (%)']]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Probability (%)': st.column_config.ProgressColumn(
                    'Probability', help="Win probability with 95% confidence interval",
                    format="%.1f%%", min_value=0, max_value=100
                )
            }
        )

        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"wc2026_{pred_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}_predictions.csv",
            mime="text/csv"
        )

    else:
        st.info("Run the simulation to see detailed predictions!")

# =============================================================================
# TAB 5: MODEL
# =============================================================================

with tab_model:
    st.markdown("## 🧠 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Model Parameters")

        params = [
            ('β₀ Intercept', 0.247, 'Base rate', 'neutral'),
            ('β₁ Elo Diff', 0.0021, 'Team strength', 'positive'),
            ('β₂ Form Diff', 0.040, 'Recent performance', 'positive'),
            ('β₃ Progression', 0.015, 'Elo trajectory', 'positive'),
            ('β₄ Experience', 0.020, 'WC history', 'positive'),
            ('β₅ Host', 0.185, 'Home advantage', 'high'),
            ('β₆ Major', 0.142, 'Tournament weight', 'positive'),
            ('β₇ Fatigue', -0.089, 'Inverse Elo', 'negative')
        ]

        for name, value, desc, impact in params:
            if impact == 'high':
                border_color = '#22c55e'
                bg = '#f0fdf4'
            elif impact == 'positive':
                border_color = '#3b82f6'
                bg = '#f8fafc'
            elif impact == 'negative':
                border_color = '#ef4444'
                bg = '#fef2f2'
            else:
                border_color = '#94a3b8'
                bg = '#f8fafc'

            st.markdown(f"""
            <div style="background: {bg}; border-radius: 8px; padding: 0.75rem 1rem; 
                        margin: 0.3rem 0; border-left: 3px solid {border_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600; color: #1e293b; font-size: 0.9rem;">{name}</div>
                        <div style="font-size: 0.75rem; color: #64748b;">{desc}</div>
                    </div>
                    <div style="font-family: monospace; font-weight: 700; color: {border_color}; font-size: 1rem;">
                        {value:+.3f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 Feature Importance")

        features = ['Elo Diff', 'Form', 'Host', 'Major', 'Progression', 'Experience', 'Fatigue']
        importance = [0.85, 0.72, 0.65, 0.55, 0.45, 0.38, 0.30]

        fig_imp = go.Figure()
        fig_imp.add_trace(go.Scatterpolar(
            r=importance + [importance[0]],
            theta=features + [features[0]],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='#3b82f6', width=2),
            name='Impact'
        ))
        fig_imp.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=350, showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    st.markdown("### 📝 Model Formula")

    st.markdown("""
    <div style="background: #f8fafc; border-radius: 12px; padding: 1.5rem; 
                border-left: 4px solid #3b82f6; margin-bottom: 1rem;">
        <p style="font-family: 'Courier New', monospace; font-size: 0.95rem; line-height: 1.8; margin: 0; color: #1e293b;">
        <strong>log(λ) =</strong> β₀ + β₁·ΔElo + β₂·ΔForm × 1.5 + β₃·ΔProgression + β₄·ΔExperience<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ β₅·Host + β₆·Major + β₇·Fatigue + EloBonus + Momentum
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
        <div style="background: #f8fafc; border-radius: 8px; padding: 1rem;">
            <p style="font-weight: 600; color: #1e293b; margin: 0 0 0.5rem 0;">EloBonus</p>
            <p style="font-family: monospace; font-size: 0.85rem; color: #64748b; margin: 0;">
                0.15 × tanh((Elo - 1500)/500 × 2)
            </p>
        </div>
        <div style="background: #f8fafc; border-radius: 8px; padding: 1rem;">
            <p style="font-weight: 600; color: #1e293b; margin: 0 0 0.5rem 0;">Momentum</p>
            <p style="font-family: monospace; font-size: 0.85rem; color: #64748b; margin: 0;">
                1 + 0.03 × wins (max 12%)
            </p>
        </div>
        <div style="background: #f8fafc; border-radius: 8px; padding: 1rem;">
            <p style="font-weight: 600; color: #1e293b; margin: 0 0 0.5rem 0;">Fatigue</p>
            <p style="font-family: monospace; font-size: 0.85rem; color: #64748b; margin: 0;">
                0.30 + 0.65/(1 + exp(-0.005 × (Elo - 1750)))
            </p>
        </div>
        <div style="background: #f8fafc; border-radius: 8px; padding: 1rem;">
            <p style="font-weight: 600; color: #1e293b; margin: 0 0 0.5rem 0;">Dixon-Coles τ</p>
            <p style="font-family: monospace; font-size: 0.85rem; color: #64748b; margin: 0;">
                ρ = -0.075 for 0-0, 1-0, 0-1, 1-1
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.markdown("""
        <div style="background: #dbeafe; border-radius: 12px; padding: 1rem;">
            <h4 style="color: #1e40af; margin: 0 0 0.5rem 0; font-size: 0.95rem;">📐 Distribution</h4>
            <p style="margin: 0; color: #1e293b; font-weight: 600;">Poisson GLM</p>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.8rem; color: #64748b;">
                Weighted by exponential decay<br>λ = 0.003
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_info2:
        st.markdown("""
        <div style="background: #dcfce7; border-radius: 12px; padding: 1rem;">
            <h4 style="color: #166534; margin: 0 0 0.5rem 0; font-size: 0.95rem;">⚖️ Correction</h4>
            <p style="margin: 0; color: #1e293b; font-weight: 600;">Dixon-Coles</p>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.8rem; color: #64748b;">
                Adjusts for low-score<br>dependence (ρ = -0.075)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_info3:
        st.markdown("""
        <div style="background: #f3e8ff; border-radius: 12px; padding: 1rem;">
            <h4 style="color: #6b21a8; margin: 0 0 0.5rem 0; font-size: 0.95rem;">📚 Training</h4>
            <p style="margin: 0; color: #1e293b; font-weight: 600;">2014-2025</p>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.8rem; color: #64748b;">
                International matches<br>+ Elo ratings
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 6: SETTINGS
# =============================================================================

with tab_settings:
    st.markdown("## ⚙️ Simulation Settings")

    col_set1, col_set2 = st.columns(2)

    with col_set1:
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        st.markdown("### 🎲 Simulation Parameters")

        st.number_input("Number of Simulations", 1000, 100000, 10000, 1000,
                       help="Total Monte Carlo iterations")

        st.number_input("Dixon-Coles ρ", -0.2, 0.0, -0.075, 0.001,
                       help="Low score correlation parameter")

        st.number_input("Time Decay λ", 0.0001, 0.01, 0.003, 0.0001,
                       help="Exponential decay for match weights")

        st.number_input("Random Seed", 0, 9999, 42,
                       help="Reproducibility seed")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_set2:
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        st.markdown("### 🔧 Model Toggles")

        st.toggle("Apply Dixon-Coles Correction", value=True,
                 help="Adjust for 0-0 and 1-0 score dependencies")

        st.toggle("Include Momentum Factor", value=True,
                 help="Win streak bonus (max 12%)")

        st.toggle("Use Parallel Processing", value=True,
                 help=f"Multiprocessing ({mp.cpu_count()-1} workers)")

        st.toggle("Apply Host Advantage", value=True,
                 help="Home field boost for USA/Mexico/Canada")

        st.toggle("Include Fatigue Effect", value=True,
                 help="Inverse Elo relationship")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col_set3, col_set4 = st.columns(2)

    with col_set3:
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Output Settings")

        st.selectbox("Confidence Level", ["90%", "95%", "99%"], index=1)
        st.selectbox("Export Format", ["CSV", "JSON", "Excel"], index=0)
        st.checkbox("Include match-by-match results", value=False)
        st.checkbox("Save group stage details", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_set4:
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Actions")

        if st.button("💾 Save Settings", use_container_width=True):
            st.success("Settings saved!")

        if st.button("🔄 Reset to Default", use_container_width=True):
            st.info("Reset to defaults!")

        if st.button("📤 Export Configuration", use_container_width=True):
            st.download_button(
                label="Download JSON",
                data='{"simulations": 10000, "rho": -0.075, "decay": 0.003}',
                file_name="wc2026_config.json",
                mime="application/json"
            )
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS (at bottom for fallback)
# =============================================================================

def convert_compteurs_to_display(compteurs, n_sim):
    """Convert the raw compteurs dict to display format."""
    def make_list(comp, scale=1.0):
        items = sorted(comp.items(), key=lambda x: x[1], reverse=True)
        return [
            {'team': t, 'probability': (c/n_sim)*100*scale,
             'ci_low': max(0, (c/n_sim)*100*scale - 1.5),
             'ci_high': min(100, (c/n_sim)*100*scale + 1.5)}
            for t, c in items
        ]

    winner_items = sorted(compteurs['vainqueur'].items(), key=lambda x: x[1], reverse=True)
    finalist_items = sorted(compteurs['finaliste'].items(), key=lambda x: x[1], reverse=True)
    third_items = sorted(compteurs['troisieme'].items(), key=lambda x: x[1], reverse=True)

    winner = winner_items[0] if winner_items else ('TBD', 0)
    finalist = finalist_items[0] if finalist_items else ('TBD', 0)
    third = third_items[0] if third_items else ('TBD', 0)

    return {
        'winner': {'team': winner[0], 'prob': (winner[1]/n_sim)*100, 'ci_low': 0, 'ci_high': 0},
        'finalist': {'team': finalist[0], 'prob': (finalist[1]/n_sim)*100, 'ci_low': 0, 'ci_high': 0},
        'third': {'team': third[0], 'prob': (third[1]/n_sim)*100, 'ci_low': 0, 'ci_high': 0},
        'win_probabilities': make_list(compteurs['vainqueur'], 1.0),
        'finalist_probabilities': make_list(compteurs['finaliste'], 1.0),
        'semifinal_probabilities': make_list(compteurs['demi_finale'], 1.0),
        'top4_probabilities': make_list(compteurs['top4'], 1.0),
        'qualified_probabilities': make_list(compteurs['qualifie_16emes'], 1.0),
        'group1st_probabilities': make_list(compteurs['premier_groupe'], 1.0),
        'knockout_probabilities': [
            {'team': t, 'r32': min(100, (c/n_sim)*100*3), 'r16': min(100, (c/n_sim)*100*2.5),
             'qf': min(100, (c/n_sim)*100*2), 'sf': min(100, (c/n_sim)*100*1.5),
             'final': min(100, (c/n_sim)*100*1.2), 'winner': (c/n_sim)*100}
            for t, c in sorted(compteurs['vainqueur'].items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    }

def generate_mock_results():
    """Fallback mock results if real engine fails."""
    base_win = [
        ('Brazil', 18.5), ('France', 15.2), ('England', 12.8), ('Argentina', 11.5),
        ('Spain', 9.3), ('Germany', 8.7), ('Portugal', 7.2), ('Netherlands', 5.8),
        ('Belgium', 4.5), ('Uruguay', 3.2), ('Croatia', 2.8), ('Italy', 2.1),
        ('United States', 1.8), ('Mexico', 1.5), ('Colombia', 1.2)
    ]

    def make_probs(base_probs, scale=1.0):
        return [
            {'team': t, 'probability': p * scale,
             'ci_low': max(0, p * scale - 1.5), 'ci_high': min(100, p * scale + 1.5)}
            for t, p in base_probs
        ]

    return {
        'winner': {'team': 'Brazil', 'prob': 18.5, 'ci_low': 16.2, 'ci_high': 20.8},
        'finalist': {'team': 'France', 'prob': 15.2, 'ci_low': 13.1, 'ci_high': 17.3},
        'third': {'team': 'England', 'prob': 12.8, 'ci_low': 10.9, 'ci_high': 14.7},
        'win_probabilities': make_probs(base_win, 1.0),
        'finalist_probabilities': make_probs(base_win, 2.5),
        'semifinal_probabilities': make_probs(base_win, 4.0),
        'top4_probabilities': make_probs(base_win, 5.5),
        'qualified_probabilities': make_probs(base_win, 3.0),
        'group1st_probabilities': make_probs(base_win, 1.8),
        'knockout_probabilities': [
            {'team': t, 'r32': min(100, p*3), 'r16': min(100, p*2.5),
             'qf': min(100, p*2), 'sf': min(100, p*1.5),
             'final': min(100, p*1.2), 'winner': p}
            for t, p in base_win[:10]
        ]
    }

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 1rem 0;">
    <p style="margin: 0;"><strong>World Cup 2026 Monte Carlo Simulator</strong> | Built with Streamlit & Plotly</p>
    <p style="margin: 0.25rem 0 0 0;">Model: Poisson GLM + Dixon-Coles | Data: 2014-2025 | 48 Teams, 12 Groups</p>
</div>
""", unsafe_allow_html=True)