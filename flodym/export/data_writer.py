import logging
import os
import pickle
from typing import Callable

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
