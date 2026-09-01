import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Competitive Solvent Extraction Simulator",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# METAL DATABASE
# ============================================================

METALS = [
    "La", "Ce", "Pr", "Nd", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho",
    "Y", "Er", "Tm", "Yb", "Lu",
    "Fe", "Co", "Ni", "Cu", "Zn",
    "Mn", "Ca", "Mg", "Al"
]

# Metal charges (valence): for rare earths and common metals in extraction
METAL_CHARGE = {
    "La": 3, "Ce": 3, "Pr": 3, "Nd": 3, "Sm": 3,
    "Eu": 3, "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3,
    "Y": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,
    "Fe": 3, "Co": 2, "Ni": 2, "Cu": 2, "Zn": 2,
    "Mn": 2, "Ca": 2, "Mg": 2, "Al": 3
}


MOLAR_MASS = {

    "La": 138.90547,
    "Ce": 140.116,
    "Pr": 140.90766,
    "Nd": 144.242,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.249,
    "Tb": 158.92535,
    "Dy": 162.500,
    "Ho": 164.93033,
    "Y": 88.90584,
    "Er": 167.259,
    "Tm": 168.93422,
    "Yb": 173.045,
    "Lu": 174.9668,

    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Mn": 54.93804,
    "Ca": 40.078,
    "Mg": 24.305,
    "Al": 26.98154
}

KEX_DATABASE = {

    "DEHPA": {

        "La": 1.0e-3,
        "Ce": 1.6e-3,
        "Pr": 2.5e-3,
        "Nd": 4.0e-3,
        "Sm": 1.2e-2,
        "Eu": 2.0e-2,
        "Gd": 3.0e-2,
        "Tb": 5.0e-2,
        "Dy": 8.0e-2,
        "Ho": 1.2e-1,
        "Y": 1.0e-1,
        "Er": 1.8e-1,
        "Tm": 2.5e-1,
        "Yb": 3.5e-1,
        "Lu": 5.0e-1,

        "Fe": 2.0e-1,
        "Co": 2.0e-3,
        "Ni": 3.0e-3,
        "Cu": 5.0e-2,
        "Zn": 1.0e-2,
        "Mn": 2.0e-3,
        "Ca": 5.0e-4,
        "Mg": 2.0e-4,
        "Al": 5.0e-2
    },

    "P507": {

        "La": 8.0e-4,
        "Ce": 1.3e-3,
        "Pr": 2.0e-3,
        "Nd": 3.2e-3,
        "Sm": 1.0e-2,
        "Eu": 1.7e-2,
        "Gd": 2.6e-2,
        "Tb": 4.5e-2,
        "Dy": 7.0e-2,
        "Ho": 1.0e-1,
        "Y": 8.5e-2,
        "Er": 1.5e-1,
        "Tm": 2.2e-1,
        "Yb": 3.0e-1,
        "Lu": 4.3e-1,

        "Fe": 1.5e-1,
        "Co": 1.5e-3,
        "Ni": 2.5e-3,
        "Cu": 4.0e-2,
        "Zn": 8.0e-3,
        "Mn": 1.5e-3,
        "Ca": 4.0e-4,
        "Mg": 1.5e-4,
        "Al": 4.0e-2
    },

    "Cyanex": {

        "La": 1.2e-3,
        "Ce": 2.0e-3,
        "Pr": 3.2e-3,
        "Nd": 5.0e-3,
        "Sm": 1.5e-2,
        "Eu": 2.5e-2,
        "Gd": 3.8e-2,
        "Tb": 6.0e-2,
        "Dy": 1.0e-1,
        "Ho": 1.5e-1,
        "Y": 1.3e-1,
        "Er": 2.2e-1,
        "Tm": 3.2e-1,
        "Yb": 4.5e-1,
        "Lu": 6.5e-1,

        "Fe": 2.5e-1,
        "Co": 1.0e-1,   # CORRIGIDO: era 2.5e-3 (Co < Ni), agora Co >> Ni
        "Ni": 4.0e-3,   # mantido
        "Cu": 7.0e-2,
        "Zn": 1.5e-2,
        "Mn": 2.5e-3,
        "Ca": 7.0e-4,
        "Mg": 3.0e-4,
        "Al": 7.0e-2
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def concentration_to_mol(value, unit, metal):
    """Convert concentration to mol/L"""
    if unit == "mol/L":
        return value

    if unit == "g/L":
        return value / MOLAR_MASS[metal]

    if unit == "mg/L":
        return (value / 1000.0) / MOLAR_MASS[metal]

    return 0.0


# ============================================================
# COMPETITIVE EQUILIBRIUM SOLVER
# ============================================================

def solve_competitive_extraction(
    feed,
    h_initial,
    extractant_total,
    oa_ratio,
    kex_values,
    metal_charges
):
    """
    Solve single-stage multicomponent equilibrium extraction.

    MODEL ASSUMPTIONS / IMPLEMENTATION NOTES
    - Stoichiometry: M^z+ + z*HA <-> MA_z + z*H+
    - Monomer/dimer convention (FIXED): DEHPA, P507 and Cyanex 272 all
      dimerize in non-polar diluents in reality. Rather than modeling a
      separate dimerization equilibrium (which would require a Kdim not
      normally available for a pre-experimental screening tool), this
      model follows the standard practical convention used when Kex is
      reported/fitted as an apparent constant on a FORMAL MONOMER basis:
      'extractant_total' and 'kex_values' are BOTH expressed per mole of
      monomeric HA, consistently, everywhere in this function. The
      previous version of this code carried an unused 'aggregation'
      parameter that implied dimer/monomer bookkeeping was happening
      when it was not -- that parameter has been removed to avoid this
      false impression. There is exactly one convention in force now
      (formal monomer basis), and it is used consistently in the
      extractant mass balance, the Kex expression, and the UI labels.
    - Saponification has been intentionally removed from this model
      (previously represented as extractant permanently withdrawn from
      the free pool). Modeling neutralized extractant correctly would
      require tracking Na+/NH4+ for the neutralized fraction instead of
      H+, which this single-stage H+-only mass balance does not attempt.
    - Activities, ionic strength, temperature and metal hydrolysis are
      neglected (by design, for a pre-experimental screening tool).

    Parameters
    ----------
    feed : array
        Initial metal concentrations (mol/L, aqueous)
    h_initial : float
        Initial H+ concentration (mol/L, aqueous)
    extractant_total : float
        Total extractant concentration (mol/L, formal monomer basis, per L_org)
    oa_ratio : float
        Organic to aqueous phase ratio (V_org / V_aq)
    kex_values : array
        Kex for each metal (basis: M^z+ + z*HA <-> MA_z + z*H+, formal monomer)
    metal_charges : array
        Charge of each metal (z value)

    Returns
    -------
    dict
        Equilibrium concentrations and extraction %
    """

    feed = np.asarray(feed, dtype=float)
    kex_values = np.asarray(kex_values, dtype=float)
    metal_charges = np.asarray(metal_charges, dtype=float)

    oa = max(float(oa_ratio), 1e-12)
    extractant_total = max(float(extractant_total), 1e-12)

    # initial guesses (formal monomer basis for extractant)
    h_guess = max(h_initial, 1e-10)
    e_guess = max(extractant_total * 0.5, 1e-12)

    def residual(log_variables):
        """
        Residual equations for equilibrium:
        1. Extractant balance: e_free = extractant_total - extracted
        2. H+ balance: h = h_initial + h_produced
        All metal concentrations are on a per-L_aq basis unless noted.
        Extractant concentrations are on a per-L_org, formal-monomer basis;
        conversions between the two bases are done explicitly where needed.
        """

        h = np.exp(log_variables[0])
        e_free = np.exp(log_variables[1])  # formal monomer basis (mol HA / L_org)

        # Distribution coefficient: Kex = [MA_z]_org * [H+]^z / ([M^z+]_aq * [HA]_org^z)
        # For dilute solutions: D_i = Kex_i * [HA]^z / [H+]^z
        D = kex_values * (e_free ** metal_charges) / (h ** metal_charges)

        # Aqueous concentrations at equilibrium (per L_aq)
        caq = feed / (1.0 + D * oa)

        # Organic concentrations at equilibrium (expressed per L_aq)
        corg_aq = D * oa * caq

        # Convert organic metal concentration to per L_org for mass balances involving
        # extractant (extractant_total is per L_org)
        corg_org = corg_aq / oa

        # Extractant consumed: z moles of HA (formal monomer) per mole of metal extracted
        extractant_consumed = np.sum(
            metal_charges[feed > 0] * corg_org[feed > 0]
        )

        # Expected free extractant (formal monomer basis, per L_org)
        e_expected = max(
            1e-12,
            extractant_total - extractant_consumed
        )

        # H+ generation: z moles per mole of metal (produced in aqueous phase per L_aq)
        h_produced = np.sum(
            metal_charges[feed > 0] * corg_aq[feed > 0]
        )

        # H+ balance: initial + produced (no external base/buffer is modelled)
        h_expected = h_initial + h_produced

        # Scaling for numerical stability
        scale_e = max(extractant_total, 1e-8)
        scale_h = max(max(h_expected, h_initial), 1e-8)

        return np.array([
            (e_free - e_expected) / scale_e,
            (h - h_expected) / scale_h
        ])

    # Solve using least_squares with log transform for positivity
    result = least_squares(
        residual,
        np.log([h_guess, e_guess]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=5000
    )

    h = np.exp(result.x[0])
    e_free = np.exp(result.x[1])

    # Recalculate at convergence
    D = kex_values * (e_free ** metal_charges) / (h ** metal_charges)

    caq = feed / (1.0 + D * oa)
    corg_aq = D * oa * caq
    corg_org = corg_aq / oa

    # Extraction percentage (only for metals present in feed)
    extraction = np.divide(
        corg_aq,
        feed,
        out=np.zeros_like(corg_aq),
        where=feed > 0
    ) * 100.0

    extraction = np.clip(extraction, 0.0, 100.0)

    extracted_total = np.sum(corg_aq)
    extractant_consumed = np.sum(
        metal_charges[feed > 0] * corg_org[feed > 0]
    )

    h_produced = np.sum(
        metal_charges[feed > 0] * corg_aq[feed > 0]
    )

    # Basic consistency checks (not raising errors, but returned for inspection)
    consistency = {
        "extractant_consumed_le_total": extractant_consumed <= (extractant_total + 1e-8),
        "free_extractant_nonnegative": e_free >= 0,
    }

    return {
        "h": h,
        "pH": -np.log10(max(h, 1e-30)),
        "free_extractant": e_free,
        "D": D,
        "caq": caq,
        "corg": corg_aq,
        "extraction": extraction,
        "extracted_total": extracted_total,
        "extractant_consumed": extractant_consumed,
        "h_produced": h_produced,
        "success": result.success,
        "consistency": consistency
    }


# ============================================================
# GRAPH FUNCTIONS
# ============================================================


def calculate_sweep(
    parameter,
    values,
    feed,
    pH,
    extractant_total,
    oa_ratio,
    kex_values,
    selected_metals,
    metal_charges
):
    """Calculate extraction % for a sweep over one parameter"""

    results = {metal: [] for metal in selected_metals}

    for value in values:

        current_pH = pH
        current_extractant = extractant_total
        current_oa = oa_ratio

        if parameter == "pH":
            current_pH = value
        elif parameter == "Extractant":
            current_extractant = value
        elif parameter == "O/A":
            current_oa = value

        result = solve_competitive_extraction(
            feed,
            10 ** (-current_pH),
            current_extractant,
            current_oa,
            kex_values,
            metal_charges
        )

        for metal in selected_metals:
            metal_index = METALS.index(metal)
            results[metal].append(result["extraction"][metal_index])

    return results


def plot_sweep(x, results, xlabel, title, fixed_text):
    """Plot extraction vs. parameter"""

    fig, ax = plt.subplots(figsize=(9, 5))

    for metal, values in results.items():
        ax.plot(
            x, values,
            marker="o",
            markersize=2.5,
            linewidth=2,
            label=metal
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Extraction (%)")
    ax.set_title(title)

    ax.text(
        0.02, 0.03,
        fixed_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8
        )
    )

    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    return fig


# ============================================================
# APPLICATION
# ============================================================

st.title("Competitive Solvent Extraction Simulator")

st.caption(
    "Single-stage multicomponent equilibrium for "
    "acidic organophosphorus extractants."
)

# ============================================================
# BASIC INPUTS
# ============================================================

st.header("1. Solution composition")

col1, col2 = st.columns(2)

with col1:
    concentration_unit = st.selectbox(
        "Concentration unit",
        ["mg/L", "g/L", "mol/L"]
    )

with col2:
    pH = st.number_input(
        "Operating pH",
        min_value=-2.0,
        max_value=14.0,
        value=2.0,
        step=0.1
    )


# ============================================================
# METAL SELECTION
# ============================================================

st.subheader("Select metals")

selected_metals = st.multiselect(
    "Metals present in the aqueous solution",
    METALS,
    default=["Nd", "Sm", "Fe"]
)

feed = np.zeros(len(METALS))

if selected_metals:

    st.subheader("Metal concentrations")

    cols = st.columns(4)

    for i, metal in enumerate(selected_metals):

        with cols[i % 4]:

            value = st.number_input(
                f"{metal} ({concentration_unit})",
                min_value=0.0,
                value=1.0,
                format="%.8g",
                key=f"conc_{metal}"
            )

            feed[METALS.index(metal)] = concentration_to_mol(
                value,
                concentration_unit,
                metal
            )

else:
    st.warning("Select at least one metal.")
    st.stop()


# ============================================================
# EXTRACTANT
# ============================================================

st.header("2. Extractant")

col1, col2 = st.columns(2)

with col1:
    extractant_type = st.selectbox(
        "Phosphorus-based extractant",
        ["DEHPA", "P507", "Cyanex"]
    )

with col2:
    extractant_total = st.number_input(
        "Total extractant concentration (mol/L, formal monomer basis)",
        min_value=0.000001,
        value=0.5,
        step=0.05
    )


# ============================================================
# O/A RATIO
# ============================================================

st.header("3. Phase ratio")

oa_ratio = st.number_input(
    "O/A ratio",
    min_value=0.001,
    value=1.0,
    step=0.1
)


# ============================================================
# DATABASE INFORMATION
# ============================================================

with st.expander("View provisional Kex values and stoichiometry"):

    kex_table = []

    for metal in selected_metals:
        charge = METAL_CHARGE[metal]
        kex_table.append({
            "Metal": metal,
            "Charge (z)": charge,
            "Stoichiometry": f"M^{charge}+ + {charge}HA -> MA_{charge} + {charge}H+",
            "Kex": KEX_DATABASE[extractant_type][metal]
        })

    st.dataframe(kex_table, use_container_width=True, hide_index=True)

# ============================================================
# KEX ARRAY AND METAL CHARGES
# ============================================================

kex_values = np.array([
    KEX_DATABASE[extractant_type].get(metal, 0.0)
    for metal in METALS
])

metal_charges = np.array([
    METAL_CHARGE.get(metal, 3)
    for metal in METALS
])


# ============================================================
# GRAPH SETTINGS
# ============================================================

st.header("4. Graph settings")

st.write(
    "For each graph, choose the variable to study. "
    "All other operating conditions remain fixed at the values selected above."
)


# ============================================================
# pH GRAPH
# ============================================================

st.subheader("Extraction vs. pH")

pH_min = st.number_input(
    "Minimum pH",
    min_value=-2.0,
    max_value=14.0,
    value=0.0,
    step=0.5,
    key="pH_min"
)

pH_max = st.number_input(
    "Maximum pH",
    min_value=-2.0,
    max_value=14.0,
    value=6.0,
    step=0.5,
    key="pH_max"
)

pH_values = np.linspace(
    min(pH_min, pH_max),
    max(pH_min, pH_max),
    60
)

pH_results = calculate_sweep(
    "pH",
    pH_values,
    feed,
    pH,
    extractant_total,
    oa_ratio,
    kex_values,
    selected_metals,
    metal_charges
)

st.pyplot(
    plot_sweep(
        pH_values,
        pH_results,
        "pH",
        "Extraction vs. pH",
        (
            f"Fixed: {extractant_type} | "
            f"Extractant = {extractant_total:.4g} M | "
            f"O/A = {oa_ratio:.4g}"
        )
    ),
    clear_figure=True
)


# ============================================================
# O/A GRAPH
# ============================================================

st.subheader("Extraction vs. O/A ratio")

ao_min = st.number_input(
    "Minimum O/A",
    min_value=0.001,
    value=0.1,
    step=0.1,
    key="ao_min"
)

ao_max = st.number_input(
    "Maximum O/A",
    min_value=0.001,
    value=10.0,
    step=0.5,
    key="ao_max"
)

ao_values = np.geomspace(
    min(ao_min, ao_max),
    max(ao_min, ao_max),
    60
)

ao_results = calculate_sweep(
    "O/A",
    ao_values,
    feed,
    pH,
    extractant_total,
    oa_ratio,
    kex_values,
    selected_metals,
    metal_charges
)

st.pyplot(
    plot_sweep(
        ao_values,
        ao_results,
        "O/A ratio",
        "Extraction vs. O/A ratio",
        (
            f"Fixed: pH = {pH:.2f} | "
            f"{extractant_type} | "
            f"Extractant = {extractant_total:.4g} M"
        )
    ),
    clear_figure=True
)


# ============================================================
# EXTRACTANT GRAPH
# ============================================================

st.subheader("Extraction vs. extractant concentration")

extractant_min = st.number_input(
    "Minimum extractant concentration (mol/L)",
    min_value=0.000001,
    value=0.01,
    step=0.01,
    key="extractant_min"
)

extractant_max = st.number_input(
    "Maximum extractant concentration (mol/L)",
    min_value=0.000001,
    value=2.0,
    step=0.1,
    key="extractant_max"
)

extractant_values = np.geomspace(
    min(extractant_min, extractant_max),
    max(extractant_min, extractant_max),
    60
)

extractant_results = calculate_sweep(
    "Extractant",
    extractant_values,
    feed,
    pH,
    extractant_total,
    oa_ratio,
    kex_values,
    selected_metals,
    metal_charges
)

st.pyplot(
    plot_sweep(
        extractant_values,
        extractant_results,
        "Extractant concentration (mol/L, formal monomer basis)",
        "Extraction vs. extractant concentration",
        (
            f"Fixed: pH = {pH:.2f} | "
            f"{extractant_type} | "
            f"O/A = {oa_ratio:.4g}"
        )
    ),
    clear_figure=True
)


# ============================================================
# CURRENT OPERATING POINT
# ============================================================

st.header("Current operating point")

base_result = solve_competitive_extraction(
    feed,
    10 ** (-pH),
    extractant_total,
    oa_ratio,
    kex_values,
    metal_charges
)

if not base_result["success"]:
    st.warning(
        "The equilibrium solver did not converge for these operating "
        "conditions. Results below may be inaccurate -- try adjusting "
        "pH, extractant concentration, or O/A ratio."
    )

summary = []

for metal in selected_metals:
    i = METALS.index(metal)
    z = METAL_CHARGE[metal]

    summary.append({
        "Metal": metal,
        "z": z,
        "Initial (mol/L)": feed[i],
        "Aqueous (mol/L)": base_result["caq"][i],
        "Organic (mol/L)": base_result["corg"][i],
        "Extraction (%)": base_result["extraction"][i],
        "Distribution coefficient": base_result["D"][i]
    })

st.dataframe(summary, use_container_width=True, hide_index=True)

# Summary statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Equilibrium pH",
        f"{base_result['pH']:.4f}"
    )

with col2:
    st.metric(
        "Free Extractant (M)",
        f"{base_result['free_extractant']:.6f}"
    )

with col3:
    st.metric(
        "H+ Produced (mol/L)",
        f"{base_result['h_produced']:.6f}"
    )

with col4:
    st.metric(
        "Extractant Used (mol/L)",
        f"{base_result['extractant_consumed']:.6f}"
    )
