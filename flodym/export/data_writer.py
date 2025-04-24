import logging
import os
import pickle
from typing import Callable, Iterable
import pandas as pd
import inspect

from ..mfa_system import MFASystem
from ..flodym_arrays import FlodymArray
from .helper import to_valid_file_name


def export_mfa_to_pickle(mfa: MFASystem, export_path: str):
    """Write an MFA system to a pickle file.

    Args:
        mfa (MFASystem): The MFA system to be exported.
        export_path (str): The path to the file where the MFA system should be saved.
    """
    dict_out = convert_to_dict(mfa)
    pickle.dump(dict_out, open(export_path, "wb"))
    logging.info(f"Data saved to {export_path}")


def export_mfa_flows_to_csv(mfa: MFASystem, export_directory: str):
    """export flows of an MFA system to csv files.

    Args:
        mfa (MFASystem): The MFA system from which the flows should be exported.
        export_directory (str): The directory where the csv files should be saved.
    """
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
    for flow_name, flow in mfa.flows.items():
        path_out = os.path.join(export_directory, f"{to_valid_file_name(flow_name)}.csv")
        flow.to_df().to_csv(path_out)
    logging.info(f"Data saved in directory {export_directory}")


def export_mfa_stocks_to_csv(mfa: MFASystem, export_directory: str, with_in_and_out: bool = False):
    """export stocks of an MFA system to csv files.

    Args:
        mfa (MFASystem): The MFA system from which the stocks should be exported.
        export_directory (str): The directory where the csv files should be saved.
        with_in_and_out (bool, optional): If True, the inflow and outflow of the stocks are also exported. Defaults to False.
    """
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
    for stock_name, stock in mfa.stocks.items():
        output_items = {"stock": stock.stock}
        if with_in_and_out:
            output_items["inflow"] = stock.inflow
            output_items["outflow"] = stock.outflow
        for attribute_name, output in output_items.items():
            df = output.to_df()
            path_out = os.path.join(
                export_directory, f"{to_valid_file_name(stock_name)}_{attribute_name}.csv"
            )
            df.to_csv(path_out, index=False)
    logging.info(f"Data saved in directory {export_directory}")


def convert_to_dict(mfa: MFASystem, type: str = "numpy") -> dict:
    """Convert an MFA system to a dictionary which is readable without flodym.

    Args:
        mfa (MFASystem): The MFA system to be converted.
        type (str, optional): The type of the values in the flows and stocks. Options are 'numpy' and 'pandas'. Defaults to "numpy".

    Returns:
        dict: The MFA system as a dictionary. Contains the items 'dimension_names', 'dimension_items', 'processes', 'flows', 'flow_dimensions', 'flow_processes', 'stocks', 'stock_dimensions', 'stock_processes'.
    """
    convert_func = _get_convert_func(type)
    return _convert_to_dict_by_func(mfa, convert_func)


def _get_convert_func(type: str):
    if type == "numpy":

        def convert_func(array: FlodymArray):
            return array.values

    elif type == "pandas":

        def convert_func(array: FlodymArray):
            return array.to_df()

    else:
        raise ValueError(f"Unknown export type {type}. Must be 'numpy' or 'pandas'.")
    return convert_func


def _convert_to_dict_by_func(mfa: MFASystem, convert_func: Callable) -> dict:
    dict_out = {}
    dict_out["dimension_names"] = {d.letter: d.name for d in mfa.dims}
    dict_out["dimension_items"] = {d.name: d.items for d in mfa.dims}
    dict_out["processes"] = [p.name for p in mfa.processes.values()]
    dict_out["flows"] = {n: convert_func(f) for n, f in mfa.flows.items()}
    dict_out["flow_dimensions"] = {n: f.dims.letters for n, f in mfa.flows.items()}
    dict_out["flow_processes"] = {
        n: (f.from_process.name, f.to_process.name) for n, f in mfa.flows.items()
    }
    dict_out["stocks"] = {s_name: convert_func(s.stock) for s_name, s in mfa.stocks.items()}
    dict_out["stock_dimensions"] = {
        s_name: s.stock.dims.letters for s_name, s in mfa.stocks.items()
    }
    dict_out["stock_processes"] = {
        s_name: s.process.name for s_name, s in mfa.stocks.items() if s.process is not None
    }
    return dict_out


def convert_array_to_iamc_df(
    array: FlodymArray,
    model_name: str,
    unit: str,
    generate_variable_name: Callable,
    scenario: str = None,
    scenario_letter: str = None,
    time_letter: str = "t",
    region_letter: str = "r",
) -> pd.DataFrame:
    """Convert a FlodymArray to a DataFrame in IAMC format.
    Args:
        array (FlodymArray): The FlodymArray to be converted.
        model_name (str): Model name.
        unit (str): Unit of the values.
        generate_variable_name (Callable): Function that generates either the variable name, or
            the IAMC variable classification hierarchy as a list of strings.
            The function can build the dimension items or the array name into the variable name.
            To do this, the function can have arguments with the same names as the dimensions,
            or an argument 'name' to access the name of the FlodymArray.
        scenario (str, optional): Scenario name. If not given, the scenario dimension letter must be given.
        scenario_letter (str, optional): Letter of the scenario dimension. If not given, the scenario name must be given.
        time_letter (str): Letter of the time dimension. Default is 't'.
        region_letter (str): Letter of the region dimension. Default is 'r'.
    Returns:
        pd.DataFrame: DataFrame in IAMC format.
    """

    if time_letter not in array.dims.letters:
        raise ValueError(f"Time dimension '{time_letter}' not found in FlodymArray dimensions.")
    if region_letter not in array.dims.letters:
        raise ValueError(f"Region dimension '{region_letter}' not found in FlodymArray dimensions.")

    df = array.to_df(index=False, dim_to_columns=array.dims[time_letter].name)

    df["Model"] = model_name
    df["Unit"] = unit
    df.rename(columns={array.dims[region_letter].name: "Region"}, inplace=True)
    df["name"] = array.name

    if (scenario is None) == (scenario_letter is None):
        raise ValueError("Either scenario or scenario_letter must be given, but not both.")
    elif scenario is not None:
        df["Scenario"] = scenario
    else:
        if scenario_letter not in array.dims.letters:
            raise ValueError(
                f"Scenario dimension '{scenario_letter}' not found in FlodymArray dimensions."
            )
        df.rename(columns={array.dims[scenario_letter].name: "Scenario"}, inplace=True)

    used_prms = _inspect_varname_func_signature(
        array, generate_variable_name, time_letter, region_letter
    )

    def row_func(row: pd.Series) -> str:
        prm_dict = row[[used_prms]].to_dict()
        output = generate_variable_name(**prm_dict)
        if isinstance(output, Iterable):
            output = "|".join(output)
        if not isinstance(output, str):
            raise ValueError(
                f"generate_variable_name must return a string or an iterable of strings. "
                f"Found type: {type(output)}."
            )
        return output

    df["Variable"] = df.apply(row_func, axis=1)

    df = df[["Model", "Scenario", "Region", "Variable", "Unit"] + array.dims[time_letter].items]

    return df


def _inspect_varname_func_signature(
    array: FlodymArray, generate_variable_name: Callable, time_letter: str, region_letter: str
) -> list[str]:
    """Inspect the signature of the generate_variable_name function, checking if the arguments
    match the requirements for exporting the FlodymArray to the IAMC format.
    """
    prms = inspect.signature(generate_variable_name).parameters.values()
    non_keyword = [
        prm
        for prm in prms
        if prm.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    if len(non_keyword) > 0:
        raise ValueError(
            f"generate_variable_name must have keyword arguments only. Found: {non_keyword}"
        )
    all_prms = [prm.name for prm in prms]
    required_prms = [prm.name for prm in prms if prm.default == prm.empty]
    available_columns = array.dims.names + ("name",)
    if any(prm not in available_columns for prm in required_prms):
        raise ValueError(
            "generate_variable_name has keyword arguments without default which are not "
            f"provided by dimensions and name of {array.__class__.__name__}: {required_prms}"
        )
    if array.dims[time_letter].name in all_prms or array.dims[region_letter].name in all_prms:
        logging.warning(
            "generate_variable_name uses time or region dimension names as keyword arguments. "
            f"This is not advised, as the IAMC format already distinguishes between time and "
            "region outside of the variable name."
        )
    used_prms = [prm for prm in all_prms if prm in available_columns]
    required_dims = [
        d.name for d in array.dims if d.letter not in (time_letter, region_letter) and d.len > 1
    ]
    if any(dim_name not in all_prms for dim_name in required_dims):
        logging.warning(
            "generate_variable_name does not distinguish between items of the following "
            f"dimensions with length > 0: {required_dims}. This may lead to duplicate variable "
            "names (Exception: If you chose sparse output and there is only one non-zero value "
            "per variable). To avoid this, you can add the dimension names to the function "
            "arguments, or sum the array over the excess dimensions before passing it to the "
            "function (caution: summing removes the array name). "
        )
    return used_prms
