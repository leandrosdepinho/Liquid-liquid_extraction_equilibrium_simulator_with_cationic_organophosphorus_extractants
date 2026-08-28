# ============================================================
# COMPETITIVE SOLVENT EXTRACTION SIMULATOR
# Single-stage multicomponent equilibrium
# ============================================================

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
#
# Kex values are provisional and intentionally editable.
# They represent a qualitative/approximate database for
# monoprotonic acidic organophosphorus extractants.
#
# The equilibrium reaction is represented as:
#
#     M(aq) + n HR(org) <=> MR_n(org) + n H+(aq)
#
# with n = 1 for this simplified model.
#
# Kex is therefore used as:
#
#     D = Kex * [E_free] / [H+]
#
# The important point is that E_free is NOT fixed.
# It is solved from the GLOBAL extractant mass balance,
# meaning all metals compete for the same extractant pool.
# ============================================================

METALS = [
    "La", "Ce", "Pr", "Nd", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho",
    "Y", "Er", "Tm", "Yb", "Lu",
    "Fe", "Co", "Ni", "Cu", "Zn",
    "Mn", "Ca", "Mg", "Al"
]


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


# ============================================================
# PROVISIONAL Kex DATABASE
# ============================================================
#
# Values are deliberately approximate placeholders.
# Replace these with literature values later.
#
# The relative trends are chemically plausible:
# acidic organophosphorus extractants generally show
# stronger extraction toward more strongly extracted metals,
# while pH strongly controls extraction.
# ============================================================

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
        "Co": 2.5e-3,
        "Ni": 4.0e-3,
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

    if unit == "mol/L":
        return value

    if unit == "g/L":
        return value / MOLAR_MASS[metal]

    if unit == "mg/L":
        return (value / 1000.0) / MOLAR_MASS[metal]

    return 0.0


def safe_percent(value):

    return np.clip(
        value * 100.0,
        0.0,
        100.0
    )


# ============================================================
# COMPETITIVE EQUILIBRIUM SOLVER
# ============================================================

def solve_competitive_extraction(
    feed,
    h_initial,
    extractant_total,
    saponification_fraction,
    ao_ratio,
    kex_values
):
    """
    Solve one complete multicomponent equilibrium.

    Every metal is present simultaneously.

    The free extractant concentration is solved globally
    from the total extractant balance.

    O/A is explicitly included:

        C_org = D * C_aq

    and, for concentration in the aqueous phase:

        C_aq =
            C_feed /
            (1 + D * O/A)

    The extracted amount consumes extractant.

    For the monoprotonic acidic extractant model:

        M + E <=> ME + H+

    one mole of extractant is consumed per mole of
    extracted metal.

    Saponification provides a finite reservoir that
    neutralizes generated H+.

    The solver simultaneously finds:

        H+
        free extractant

    while all metal balances are satisfied.
    """

    feed = np.asarray(
        feed,
        dtype=float
    )

    kex_values = np.asarray(
        kex_values,
        dtype=float
    )

    oa = max(
        float(ao_ratio),
        1e-12
    )

    extractant_total = max(
        float(extractant_total),
        1e-12
    )

    sap_capacity = (
        extractant_total
        * saponification_fraction
    )

    # --------------------------------------------------------
    # Initial guesses
    # --------------------------------------------------------

    h_guess = max(
        h_initial,
        1e-10
    )

    e_guess = max(
        extractant_total * 0.5,
        1e-10
    )

    # --------------------------------------------------------
    # Residual equations
    # --------------------------------------------------------

    def residual(log_variables):

        h = np.exp(
            log_variables[0]
        )

        e_free = np.exp(
            log_variables[1]
        )

        # Distribution coefficients
        D = (
            kex_values
            * e_free
            / h
        )

        # A/O equilibrium
        #
        # C_org / C_aq = D * O/A
        #
        caq = (
            feed
            /
            (
                1.0
                + D * oa
            )
        )

        corg = (
            D
            * oa
            * caq
        )

        extracted = np.sum(
            corg
        )

        # ----------------------------------------------------
        # Global extractant balance
        # ----------------------------------------------------

        e_expected = max(
            1e-12,
            extractant_total
            - extracted
        )

        # ----------------------------------------------------
        # H+ generation
        # ----------------------------------------------------

        h_generated = extracted

        neutralized = min(
            h_generated,
            sap_capacity
        )

        h_expected = (
            h_initial
            + h_generated
            - neutralized
        )

        # Numerical scaling
        scale_e = max(
            extractant_total,
            1e-8
        )

        scale_h = max(
            h_expected,
            1e-8
        )

        return np.array([

            (
                e_free
                - e_expected
            ) / scale_e,

            (
                h
                - h_expected
            ) / scale_h

        ])

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    result = least_squares(
        residual,
        np.log([
            h_guess,
            e_guess
        ]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=5000
    )

    h = np.exp(
        result.x[0]
    )

    e_free = np.exp(
        result.x[1]
    )

    # --------------------------------------------------------
    # Final equilibrium
    # --------------------------------------------------------

    D = (
        kex_values
        * e_free
        / h
    )

    caq = (
        feed
        /
        (
            1.0
            + D * oa
        )
    )

    corg = (
        D
        * oa
        * caq
    )

    extraction = np.divide(
        corg,
        feed,
        out=np.zeros_like(corg),
        where=feed > 0
    ) * 100.0

    extraction = np.clip(
        extraction,
        0.0,
        100.0
    )

    extracted_total = np.sum(
        corg
    )

    neutralized = min(
        extracted_total,
        sap_capacity
    )

    sap_remaining = max(
        0.0,
        sap_capacity
        - neutralized
    )

    return {

        "h": h,

        "pH":
            -np.log10(
                max(
                    h,
                    1e-30
                )
            ),

        "free_extractant":
            e_free,

        "D":
            D,

        "caq":
            caq,

        "corg":
            corg,

        "extraction":
            extraction,

        "extracted_total":
            extracted_total,

        "sap_remaining":
            sap_remaining,

        "success":
            result.success

    }


# ============================================================
# GRAPH GENERATION
# ============================================================

def make_parameter_range(
    center,
    minimum,
    maximum,
    points=60
):

    center = max(
        center,
        minimum
    )

    center = min(
        center,
        maximum
    )

    lower = max(
        minimum,
        center / 10.0
    )

    upper = min(
        maximum,
        center * 10.0
    )

    return np.geomspace(
        lower,
        upper,
        points
    )


def calculate_sweep(
    parameter,
    values,
    feed,
    h_initial,
    extractant_total,
    saponification,
    ao_ratio,
    kex_values
):

    results = {
        metal: []
        for metal in selected_metals_global
    }

    for value in values:

        current_pH = h_initial
        current_extractant = extractant_total
        current_saponification = saponification
        current_ao = ao_ratio

        if parameter == "pH":

            current_pH = value

        elif parameter == "Extractant":

            current_extractant = value

        elif parameter == "Saponification":

            current_saponification = value

        elif parameter == "A/O":

            current_ao = value

        result = solve_competitive_extraction(

            feed,

            10 ** (-current_pH),

            current_extractant,

            current_saponification,

            current_ao,

            kex_values

        )

        for i, metal in enumerate(
            selected_metals_global
        ):

            metal_index = (
                METALS.index(metal)
            )

            results[metal].append(
                result[
                    "extraction"
                ][metal_index]
            )

    return results


def plot_sweep(
    x,
    results,
    xlabel,
    title
):

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for metal, values in results.items():

        ax.plot(
            x,
            values,
            marker="o",
            markersize=2.5,
            linewidth=2,
            label=metal
        )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        "Extraction (%)"
    )

    ax.set_title(
        title
    )

    ax.set_ylim(
        0,
        100
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
# APPLICATION
# ============================================================

st.title(
    "Competitive Solvent Extraction Simulator"
)

st.caption(
    "Single-stage multicomponent equilibrium for "
    "acidic organophosphorus extractants."
)

st.info(
    "All selected metals are solved simultaneously. "
    "They compete for the same finite extractant pool."
)


# ============================================================
# BASIC INPUTS
# ============================================================

st.header(
    "1. Solution composition"
)

col1, col2, col3 = st.columns(3)

with col1:

    concentration_unit = st.selectbox(
        "Concentration unit",
        [
            "mg/L",
            "g/L",
            "mol/L"
        ]
    )

with col2:

    initial_pH = st.number_input(
        "Initial pH",
        min_value=-2.0,
        max_value=14.0,
        value=2.0,
        step=0.1
    )

with col3:

    oa_ratio = st.number_input(
        "O/A ratio",
        min_value=0.001,
        value=1.0,
        step=0.1
    )


# ============================================================
# METAL SELECTION
# ============================================================

st.subheader(
    "Select metals"
)

selected_metals_global = st.multiselect(
    "Metals present in the aqueous solution",
    METALS,
    default=["Nd", "Sm", "Fe"]
)


feed = np.zeros(
    len(METALS)
)


if selected_metals_global:

    st.subheader(
        "Metal concentrations"
    )

    cols = st.columns(4)

    for i, metal in enumerate(
        selected_metals_global
    ):

        with cols[i % 4]:

            value = st.number_input(
                f"{metal} ({concentration_unit})",
                min_value=0.0,
                value=1.0,
                format="%.8g",
                key=f"conc_{metal}"
            )

            feed[
                METALS.index(metal)
            ] = concentration_to_mol(
                value,
                concentration_unit,
                metal
            )

else:

    st.warning(
        "Select at least one metal."
    )

    st.stop()


# ============================================================
# EXTRACTANT
# ============================================================

st.header(
    "2. Extractant"
)

col1, col2, col3 = st.columns(3)

with col1:

    extractant_type = st.selectbox(
        "Phosphorus-based extractant",
        [
            "DEHPA",
            "P507",
            "Cyanex"
        ]
    )

with col2:

    extractant_total = st.number_input(
        "Total extractant concentration "
        "(mol/L, monomer basis)",
        min_value=0.000001,
        value=0.5,
        step=0.05
    )

with col3:

    saponification_percent = st.number_input(
        "Saponification (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=5.0
    )


saponification_fraction = (
    saponification_percent
    / 100.0
)


# ============================================================
# DATABASE INFORMATION
# ============================================================

with st.expander(
    "View provisional Kex values"
):

    kex_table = []

    for metal in selected_metals_global:

        kex_table.append({

            "Metal":
                metal,

            "Kex":
                KEX_DATABASE[
                    extractant_type
                ][metal]

        })

    st.dataframe(
        kex_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The current Kex values are provisional placeholders "
        "and should be replaced by literature values."
    )


# ============================================================
# SOLVE BASE CASE
# ============================================================

kex_values = np.array([

    KEX_DATABASE[
        extractant_type
    ].get(
        metal,
        0.0
    )

    for metal in METALS

])


if st.button(
    "Calculate extraction curves",
    type="primary"
):

    st.session_state[
        "calculate_curves"
    ] = True


if st.session_state.get(
    "calculate_curves",
    False
):

    st.header(
        "3. Extraction curves"
    )

    # --------------------------------------------------------
    # pH
    # --------------------------------------------------------

    pH_values = np.linspace(
        max(
            -1,
            initial_pH - 3
        ),
        min(
            14,
            initial_pH + 3
        ),
        60
    )

    pH_results = calculate_sweep(
        "pH",
        pH_values,
        feed,
        initial_pH,
        extractant_total,
        saponification_fraction,
        oa_ratio,
        kex_values
    )

    st.pyplot(
        plot_sweep(
            pH_values,
            pH_results,
            "pH",
            "Extraction vs. pH"
        ),
        clear_figure=True
    )


    # --------------------------------------------------------
    # O/A
    # --------------------------------------------------------

    ao_values = make_parameter_range(
        oa_ratio,
        0.01,
        100.0
    )

    ao_results = calculate_sweep(
        "A/O",
        ao_values,
        feed,
        initial_pH,
        extractant_total,
        saponification_fraction,
        oa_ratio,
        kex_values
    )

    st.pyplot(
        plot_sweep(
            ao_values,
            ao_results,
            "O/A ratio",
            "Extraction vs. O/A ratio"
        ),
        clear_figure=True
    )


    # --------------------------------------------------------
    # EXTRACTANT
    # --------------------------------------------------------

    extractant_values = make_parameter_range(
        extractant_total,
        1e-4,
        10.0
    )

    extractant_results = calculate_sweep(
        "Extractant",
        extractant_values,
        feed,
        initial_pH,
        extractant_total,
        saponification_fraction,
        oa_ratio,
        kex_values
    )

    st.pyplot(
        plot_sweep(
            extractant_values,
            extractant_results,
            "Total extractant concentration (mol/L)",
            "Extraction vs. extractant concentration"
        ),
        clear_figure=True
    )


    # --------------------------------------------------------
    # SAPONIFICATION
    # --------------------------------------------------------

    sap_values = np.linspace(
        0,
        100,
        60
    )

    sap_results = calculate_sweep(
        "Saponification",
        sap_values,
        feed,
        initial_pH,
        extractant_total,
        saponification_fraction,
        oa_ratio,
        kex_values
    )

    st.pyplot(
        plot_sweep(
            sap_values,
            sap_results,
            "Saponification (%)",
            "Extraction vs. saponification"
        ),
        clear_figure=True
    )


# ============================================================
# BASE CASE RESULT
# ============================================================

st.header(
    "Current operating point"
)

base_result = solve_competitive_extraction(

    feed,

    10 ** (-initial_pH),

    extractant_total,

    saponification_fraction,

    oa_ratio,

    kex_values

)


summary = []

for metal in selected_metals_global:

    i = METALS.index(
        metal
    )

    summary.append({

        "Metal":
            metal,

        "Initial (mol/L)":
            feed[i],

        "Aqueous (mol/L)":
            base_result["caq"][i],

        "Organic (mol/L)":
            base_result["corg"][i],

        "Extraction (%)":
            base_result["extraction"][i],

        "Distribution coefficient":
            base_result["D"][i]

    })


st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


st.caption(
    f"Equilibrium pH: "
    f"{base_result['pH']:.4f} | "
    f"Free extractant: "
    f"{base_result['free_extractant']:.6f} mol/L "
    f"(monomer basis)"
)
