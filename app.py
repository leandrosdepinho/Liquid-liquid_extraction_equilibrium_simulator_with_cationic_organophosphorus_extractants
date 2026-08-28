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
    oa_ratio,
    kex_values
):

    feed = np.asarray(
        feed,
        dtype=float
    )

    kex_values = np.asarray(
        kex_values,
        dtype=float
    )

    oa = max(
        float(oa_ratio),
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

    h_guess = max(
        h_initial,
        1e-10
    )

    e_guess = max(
        extractant_total * 0.5,
        1e-10
    )

    def residual(log_variables):

        h = np.exp(
            log_variables[0]
        )

        e_free = np.exp(
            log_variables[1]
        )

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

        extracted = np.sum(
            corg
        )

        e_expected = max(
            1e-12,
            extractant_total
            - extracted
        )

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
# GRAPH FUNCTIONS
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
    pH,
    extractant_total,
    saponification,
    oa_ratio,
    kex_values,
    selected_metals
):

    results = {
        metal: []
        for metal in selected_metals
    }

    for value in values:

        current_pH = pH
        current_extractant = extractant_total
        current_saponification = saponification
        current_ao = oa_ratio

        if parameter == "pH":

            current_pH = value

        elif parameter == "Extractant":

            current_extractant = value

        elif parameter == "Saponification":

            current_saponification = value

        elif parameter == "O/A":

            current_ao = value

        result = solve_competitive_extraction(

            feed,

            10 ** (-current_pH),

            current_extractant,

            current_saponification,

            current_ao,

            kex_values

        )

        for metal in selected_metals:

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
    title,
    fixed_text
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

    ax.text(
        0.02,
        0.03,
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

col1, col2 = st.columns(2)

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

st.subheader(
    "Select metals"
)

selected_metals = st.multiselect(
    "Metals present in the aqueous solution",
    METALS,
    default=["Nd", "Sm", "Fe"]
)

feed = np.zeros(
    len(METALS)
)

if selected_metals:

    st.subheader(
        "Metal concentrations"
    )

    cols = st.columns(4)

    for i, metal in enumerate(
        selected_metals
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
# O/A
# ============================================================

st.header(
    "3. Phase ratio"
)

oa_ratio = st.number_input(
    "O/A ratio",
    min_value=0.001,
    value=1.0,
    step=0.1
)


# ============================================================
# DATABASE INFORMATION
# ============================================================

with st.expander(
    "View provisional Kex values"
):

    kex_table = []

    for metal in selected_metals:

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
# KEX ARRAY
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


# ============================================================
# GRAPH SETTINGS
# ============================================================

st.header(
    "4. Graph settings"
)

st.write(
    "For each graph, choose the variable to study. "
    "All other operating conditions remain fixed at "
    "the values selected above."
)


# ============================================================
# pH GRAPH
# ============================================================

st.subheader(
    "Extraction vs. pH"
)

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
    saponification_fraction,
    oa_ratio,
    kex_values,
    selected_metals
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
            f"O/A = {oa_ratio:.4g} | "
            f"Saponification = {saponification_percent:.1f}%"
        )
    ),
    clear_figure=True
)


# ============================================================
# O/A GRAPH
# ============================================================

st.subheader(
    "Extraction vs. O/A ratio"
)

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
    saponification_fraction,
    oa_ratio,
    kex_values,
    selected_metals
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
            f"Extractant = {extractant_total:.4g} M | "
            f"Saponification = {saponification_percent:.1f}%"
        )
    ),
    clear_figure=True
)


# ============================================================
# EXTRACTANT GRAPH
# ============================================================

st.subheader(
    "Extraction vs. extractant concentration"
)

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
    min(
        extractant_min,
        extractant_max
    ),
    max(
        extractant_min,
        extractant_max
    ),
    60
)

extractant_results = calculate_sweep(
    "Extractant",
    extractant_values,
    feed,
    pH,
    extractant_total,
    saponification_fraction,
    oa_ratio,
    kex_values,
    selected_metals
)

st.pyplot(
    plot_sweep(
        extractant_values,
        extractant_results,
        "Extractant concentration (mol/L, monomer basis)",
        "Extraction vs. extractant concentration",
        (
            f"Fixed: pH = {pH:.2f} | "
            f"{extractant_type} | "
            f"O/A = {oa_ratio:.4g} | "
            f"Saponification = {saponification_percent:.1f}%"
        )
    ),
    clear_figure=True
)


# ============================================================
# SAPONIFICATION GRAPH
# ============================================================

st.subheader(
    "Extraction vs. saponification"
)

sap_values = np.linspace(
    0,
    100,
    60
)

sap_results = calculate_sweep(
    "Saponification",
    sap_values,
    feed,
    pH,
    extractant_total,
    saponification_fraction,
    oa_ratio,
    kex_values,
    selected_metals
)

st.pyplot(
    plot_sweep(
        sap_values,
        sap_results,
        "Saponification (%)",
        "Extraction vs. saponification",
        (
            f"Fixed: pH = {pH:.2f} | "
            f"{extractant_type} | "
            f"Extractant = {extractant_total:.4g} M | "
            f"O/A = {oa_ratio:.4g}"
        )
    ),
    clear_figure=True
)


# ============================================================
# CURRENT OPERATING POINT
# ============================================================

st.header(
    "Current operating point"
)

base_result = solve_competitive_extraction(

    feed,

    10 ** (-pH),

    extractant_total,

    saponification_fraction,

    oa_ratio,

    kex_values

)


summary = []

for metal in selected_metals:

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
