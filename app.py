```python
# ============================================================
# SINGLE-STAGE METAL SOLVENT EXTRACTION SIMULATOR
# Streamlit application
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Single-Stage Solvent Extraction Simulator",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

METALS = [
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Y", "Er", "Tm", "Yb", "Lu",
    "Fe", "Co", "Ni", "Cu", "Zn", "Mn", "Ca", "Mg"
]


# ------------------------------------------------------------
# Approximate / provisional extraction constants.
#
# IMPORTANT:
# These values are placeholders intended for simulation
# development. Replace them with literature values later.
#
# The model assumes:
#
#   M^(z+) + z HL(org) ⇌ ML_z(org) + z H+
#
# For this simulator, the phosphorous extractants are treated
# as monoprotic acidic extractants.
# ------------------------------------------------------------

KEX = {

    "DEHPA": {

        "La":  2.0e-3,
        "Ce":  3.5e-3,
        "Pr":  5.0e-3,
        "Nd":  7.0e-3,
        "Sm":  1.5e-2,
        "Eu":  2.5e-2,
        "Gd":  4.0e-2,
        "Tb":  6.0e-2,
        "Dy":  8.0e-2,
        "Ho":  1.1e-1,
        "Y":   1.3e-1,
        "Er":  1.5e-1,
        "Tm":  1.8e-1,
        "Yb":  2.1e-1,
        "Lu":  2.5e-1,

        "Fe":  2.0e-1,
        "Co":  8.0e-3,
        "Ni":  1.0e-2,
        "Cu":  2.5e-2,
        "Zn":  1.8e-2,
        "Mn":  3.0e-3,
        "Ca":  4.0e-4,
        "Mg":  1.5e-4,
    },

    "P507": {

        "La":  3.0e-3,
        "Ce":  5.0e-3,
        "Pr":  7.0e-3,
        "Nd":  1.0e-2,
        "Sm":  2.2e-2,
        "Eu":  3.5e-2,
        "Gd":  5.5e-2,
        "Tb":  8.0e-2,
        "Dy":  1.1e-1,
        "Ho":  1.5e-1,
        "Y":   1.7e-1,
        "Er":  2.0e-1,
        "Tm":  2.4e-1,
        "Yb":  2.8e-1,
        "Lu":  3.2e-1,

        "Fe":  2.5e-1,
        "Co":  1.2e-2,
        "Ni":  1.5e-2,
        "Cu":  3.5e-2,
        "Zn":  2.5e-2,
        "Mn":  4.0e-3,
        "Ca":  6.0e-4,
        "Mg":  2.0e-4,
    },

    "Cyanex 272": {

        "La":  1.0e-3,
        "Ce":  1.8e-3,
        "Pr":  2.8e-3,
        "Nd":  4.0e-3,
        "Sm":  9.0e-3,
        "Eu":  1.6e-2,
        "Gd":  2.8e-2,
        "Tb":  4.2e-2,
        "Dy":  6.0e-2,
        "Ho":  8.0e-2,
        "Y":   9.0e-2,
        "Er":  1.1e-1,
        "Tm":  1.3e-1,
        "Yb":  1.5e-1,
        "Lu":  1.7e-1,

        "Fe":  1.0e-1,
        "Co":  1.8e-2,
        "Ni":  2.2e-2,
        "Cu":  2.0e-2,
        "Zn":  3.0e-2,
        "Mn":  8.0e-3,
        "Ca":  8.0e-4,
        "Mg":  3.0e-4,
    }
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
    "Mn": 54.938,
    "Ca": 40.078,
    "Mg": 24.305
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def concentration_to_mol(value, unit, metal):

    if unit == "mol/L":
        return value

    if unit == "g/L":
        return value / MOLAR_MASS[metal]

    if unit == "mg/L":
        return (value / 1000.0) / MOLAR_MASS[metal]

    return 0.0


def safe_percent(value):
    return np.clip(value * 100.0, 0.0, 100.0)


# ============================================================
# EQUILIBRIUM MODEL
# ============================================================

def calculate_D(
    metal,
    h,
    free_extractant,
    extractant_type
):

    h = max(h, 1e-30)
    free_extractant = max(
        free_extractant,
        1e-30
    )

    k = KEX[
        extractant_type
    ][metal]

    return k * (
        free_extractant ** 3
    ) / (
        h ** 3
    )


def solve_single_stage(
    concentrations,
    pH,
    ao_ratio,
    extractant_total,
    saponification,
    extractant_type
):
    """
    Single-stage equilibrium calculation.

    concentrations:
        aqueous feed concentrations in mol/L

    ao_ratio:
        A/O = aqueous volume / organic volume

    The organic phase volume is therefore:

        V_org / V_aq = 1 / A/O

    Extractant concentration is expressed on a
    monomer basis.

    Saponification is also expressed relative to
    the monomer concentration.
    """

    h_in = 10 ** (-pH)

    sap_capacity = (
        extractant_total
        * saponification
        / 100.0
    )

    # Convert A/O into O/A.
    oa_ratio = 1.0 / max(
        ao_ratio,
        1e-12
    )

    total_metal = np.sum(
        concentrations
    )

    initial_extractant_free = max(
        extractant_total
        - 3.0 * total_metal * 0.1,
        extractant_total * 0.5
    )

    def residual(log_vars):

        h_eq = np.exp(
            log_vars[0]
        )

        e_free = np.exp(
            log_vars[1]
        )

        D = np.array([
            calculate_D(
                metal,
                h_eq,
                e_free,
                extractant_type
            )
            for metal in METALS
        ])

        # A/O formulation:
        #
        # D = C_org / C_aq
        #
        # Material balance:
        #
        # C_aq,in =
        # C_aq,out +
        # C_org,out * (V_org/V_aq)
        #
        # therefore:
        #
        # C_aq,out =
        # C_aq,in /
        # [1 + D/OA]
        #
        # where OA = O/A.
        caq = concentrations / (
            1.0 + D * oa_ratio
        )

        corg = D * caq

        extracted = np.sum(
            corg * oa_ratio
        )

        # Three extractant molecules per
        # metal ion in this model.
        extractant_consumed = (
            3.0 * extracted
        )

        e_balance = (
            e_free
            - (
                extractant_total
                - extractant_consumed
            )
        )

        # H+ generated by extraction.
        h_generated = (
            3.0 * extracted
        )

        neutralized = min(
            h_generated,
            sap_capacity
        )

        h_expected = (
            h_in
            + h_generated
            - neutralized
        )

        h_balance = (
            h_eq
            - h_expected
        )

        return [

            e_balance /
            max(
                extractant_total,
                1e-8
            ),

            h_balance /
            max(
                h_expected,
                1e-8
            )
        ]

    result = least_squares(
        residual,
        np.log([
            max(h_in, 1e-8),
            initial_extractant_free
        ]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=3000
    )

    h_eq = np.exp(
        result.x[0]
    )

    e_free = np.exp(
        result.x[1]
    )

    D = np.array([
        calculate_D(
            metal,
            h_eq,
            e_free,
            extractant_type
        )
        for metal in METALS
    ])

    caq = concentrations / (
        1.0 + D * oa_ratio
    )

    corg = D * caq

    extraction = np.divide(
        concentrations - caq,
        concentrations,
        out=np.zeros_like(
            concentrations
        ),
        where=concentrations > 0
    )

    return {
        "caq": caq,
        "corg": corg,
        "D": D,
        "extraction": extraction,
        "h": h_eq,
        "pH": -np.log10(
            max(
                h_eq,
                1e-30
            )
        ),
        "free_extractant": e_free
    }


# ============================================================
# CALCULATE CURVE
# ============================================================

def calculate_curve(
    concentrations,
    parameter,
    values,
    base_pH,
    base_ao,
    base_extractant,
    base_saponification,
    extractant_type
):

    curves = {}

    for metal_index, metal in enumerate(
        METALS
    ):

        if concentrations[
            metal_index
        ] <= 0:

            continue

        curves[metal] = []

    for value in values:

        pH = base_pH
        ao = base_ao
        extractant = base_extractant
        sap = base_saponification

        if parameter == "pH":
            pH = value

        elif parameter == "A/O":
            ao = value

        elif parameter == "Extractant":
            extractant = value

        elif parameter == "Saponification":
            sap = value

        result = solve_single_stage(
            concentrations,
            pH,
            ao,
            extractant,
            sap,
            extractant_type
        )

        for i, metal in enumerate(
            METALS
        ):

            if metal in curves:

                curves[metal].append(
                    safe_percent(
                        result[
                            "extraction"
                        ][i]
                    )
                )

    return curves


# ============================================================
# PLOT
# ============================================================

def plot_parameter_curve(
    values,
    curves,
    parameter,
    operation_value
):

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    for metal, extraction in curves.items():

        ax.plot(
            values,
            extraction,
            marker="o",
            markersize=3,
            linewidth=2,
            label=metal
        )

    # Highlight the selected operating point.
    if operation_value is not None:

        ax.axvline(
            operation_value,
            linestyle="--",
            alpha=0.5
        )

    ax.set_ylim(
        0,
        100
    )

    ax.set_xlabel(
        parameter
    )

    ax.set_ylabel(
        "Extraction (%)"
    )

    ax.set_title(
        f"Metal extraction vs. {parameter}"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    fig.tight_layout()

    return fig


# ============================================================
# INTERFACE
# ============================================================

st.title(
    "Single-Stage Solvent Extraction Simulator"
)

st.caption(
    "Explore the effect of pH, A/O ratio, extractant "
    "concentration and saponification on single-stage "
    "metal extraction."
)


# ============================================================
# EXTRACTANT
# ============================================================

st.header(
    "Extractant"
)

extractant_type = st.selectbox(
    "Phosphorus-based acidic extractant",
    [
        "DEHPA",
        "P507",
        "Cyanex 272"
    ]
)

st.info(
    "The current model treats all three extractants as "
    "monoprotic acidic organophosphorus extractants. "
    "The Kex database is provisional and can be replaced "
    "with literature values later."
)


# ============================================================
# METALS
# ============================================================

st.header(
    "Aqueous feed"
)

concentration_unit = st.selectbox(
    "Concentration unit",
    [
        "mg/L",
        "g/L",
        "mol/L"
    ]
)

selected_metals = st.multiselect(
    "Select metals",
    METALS,
    default=[
        "La",
        "Nd",
        "Sm",
        "Fe",
        "Ni",
        "Cu"
    ]
)

if not selected_metals:

    st.warning(
        "Select at least one metal."
    )

    st.stop()


feed_values = {}

cols = st.columns(4)

for i, metal in enumerate(
    selected_metals
):

    with cols[
        i % 4
    ]:

        feed_values[metal] = st.number_input(
            f"{metal} ({concentration_unit})",
            min_value=0.0,
            value=1.0,
            format="%.8g",
            key=f"single_feed_{metal}"
        )


concentrations = np.zeros(
    len(METALS)
)

for metal in selected_metals:

    concentrations[
        METALS.index(metal)
    ] = concentration_to_mol(
        feed_values[metal],
        concentration_unit,
        metal
    )


# ============================================================
# OPERATING CONDITIONS
# ============================================================

st.header(
    "Operating conditions"
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    operating_pH = st.number_input(
        "Operating pH",
        min_value=-2.0,
        max_value=14.0,
        value=1.0,
        step=0.1
    )

with c2:

    operating_ao = st.number_input(
        "A/O ratio",
        min_value=0.001,
        value=1.0,
        step=0.1
    )

with c3:

    operating_extractant = st.number_input(
        "Extractant concentration "
        "(mol/L, monomer basis)",
        min_value=1e-6,
        value=0.5,
        step=0.05
    )

with c4:

    operating_saponification = st.number_input(
        "Saponification (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=5.0
    )


# ============================================================
# CURRENT RESULT
# ============================================================

st.divider()

current_result = solve_single_stage(
    concentrations,
    operating_pH,
    operating_ao,
    operating_extractant,
    operating_saponification,
    extractant_type
)

st.subheader(
    "Current operating point"
)

result_cols = st.columns(
    len(selected_metals)
)

for i, metal in enumerate(
    selected_metals
):

    idx = METALS.index(
        metal
    )

    with result_cols[
        i % len(result_cols)
    ]:

        st.metric(
            metal,
            f"{safe_percent(current_result['extraction'][idx]):.2f}%"
        )


# ============================================================
# GRAPHS
# ============================================================

st.divider()

st.header(
    "Extraction response curves"
)


# ------------------------------------------------------------
# pH
# ------------------------------------------------------------

pH_values = np.linspace(
    0.0,
    7.0,
    50
)

pH_curves = calculate_curve(
    concentrations,
    "pH",
    pH_values,
    operating_pH,
    operating_ao,
    operating_extractant,
    operating_saponification,
    extractant_type
)

st.pyplot(
    plot_parameter_curve(
        pH_values,
        pH_curves,
        "pH",
        operating_pH
    ),
    clear_figure=True
)


# ------------------------------------------------------------
# A/O
# ------------------------------------------------------------

ao_values = np.logspace(
    -2,
    2,
    50
)

ao_curves = calculate_curve(
    concentrations,
    "A/O",
    ao_values,
    operating_pH,
    operating_ao,
    operating_extractant,
    operating_saponification,
    extractant_type
)

st.pyplot(
    plot_parameter_curve(
        ao_values,
        ao_curves,
        "A/O",
        operating_ao
    ),
    clear_figure=True
)


# ------------------------------------------------------------
# EXTRACTANT CONCENTRATION
# ------------------------------------------------------------

extractant_values = np.linspace(
    max(
        operating_extractant * 0.05,
        0.001
    ),
    max(
        operating_extractant * 3.0,
        0.1
    ),
    50
)

extractant_curves = calculate_curve(
    concentrations,
    "Extractant",
    extractant_values,
    operating_pH,
    operating_ao,
    operating_extractant,
    operating_saponification,
    extractant_type
)

st.pyplot(
    plot_parameter_curve(
        extractant_values,
        extractant_curves,
        "Extractant concentration (mol/L)",
        operating_extractant
    ),
    clear_figure=True
)


# ------------------------------------------------------------
# SAPONIFICATION
# ------------------------------------------------------------

saponification_values = np.linspace(
    0,
    100,
    50
)

saponification_curves = calculate_curve(
    concentrations,
    "Saponification",
    saponification_values,
    operating_pH,
    operating_ao,
    operating_extractant,
    operating_saponification,
    extractant_type
)

st.pyplot(
    plot_parameter_curve(
        saponification_values,
        saponification_curves,
        "Saponification (%)",
        operating_saponification
    ),
    clear_figure=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Model note: Kex values are provisional placeholders "
    "for development and should be replaced by validated "
    "literature data before quantitative use."
)
```
