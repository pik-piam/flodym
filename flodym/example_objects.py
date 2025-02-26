"""
ODYM example by Stefan Pauliuk, adapted for flodym.
Same as example2.
"""

import numpy as np

from flodym import (
    Dimension,
    DimensionSet,
    FlodymArray,
    Parameter,
    FlowDefinition,
    StockDefinition,
    MFASystem,
    SimpleFlowDrivenStock,
    make_empty_flows,
    make_empty_stocks,
    make_processes,
)


class ExampleMFA(MFASystem):
    def compute(self):
        self.flows["sysenv => shredder"][...] = (
            self.parameters["eol machines"] * self.parameters["composition eol machines"]
        )
        self.flows["sysenv => demolition"][...] = (
            self.parameters["eol buildings"] * self.parameters["composition eol buildings"]
        )
        self.flows["shredder => remelting"][...] = (
            self.flows["sysenv => shredder"] * self.parameters["shredder yield"]
        )
        self.flows["shredder => sysenv"][...] = self.flows["sysenv => shredder"] * (
            1 - self.parameters["shredder yield"]
        )
        self.flows["demolition => remelting"][...] = (
            self.flows["sysenv => demolition"] * self.parameters["demolition yield"]
        )
        self.flows["demolition => landfills"][...] = self.flows["sysenv => demolition"] * (
            1 - self.parameters["demolition yield"]
        )
        self.flows["remelting => sysenv"][...] = (
            self.flows["shredder => remelting"] + self.flows["demolition => remelting"]
        ) * self.parameters["remelting yield"]
        self.flows["remelting => slag piles"][...] = (
            self.flows["shredder => remelting"] + self.flows["demolition => remelting"]
        ) * (1 - self.parameters["remelting yield"])
        self.stocks["landfills"].inflow[...] = self.flows["demolition => landfills"]
        self.stocks["landfills"].compute()
        self.stocks["slag piles"].inflow[...] = self.flows["shredder => remelting"]
        self.stocks["slag piles"].compute()


def _get_dims() -> DimensionSet:
    return DimensionSet(
        dim_list=[
            Dimension(letter="t", name="Time", dtype=int, items=list(range(1980, 2011))),
            Dimension(letter="e", name="Material", dtype=str, items=["Fe", "Cu", "Mn"]),
        ]
    )


def get_example_array() -> FlodymArray:
    dims = _get_dims()
    return FlodymArray(dims=dims, name="Example Array", values=np.random.random((31, 3)))


def get_example_mfa() -> ExampleMFA:
    dims = _get_dims()

    parameters = {
        "eol machines": Parameter(
            name="eol machines", dims=dims[("t",)], values=np.arange(15.5, 31.0, 0.5)
        ),
        "eol buildings": Parameter(
            name="eol buildings", dims=dims[("t",)], values=np.arange(120, 368, 8)
        ),
        "composition eol machines": Parameter(
            name="composition eol machines", dims=dims[("e",)], values=np.array([0.8, 0.15, 0.05])
        ),
        "composition eol buildings": Parameter(
            name="composition eol buildings",
            dims=dims[("e",)],
            values=np.array([0.95, 0.045, 0.005]),
        ),
        "shredder yield": Parameter(
            name="shredder yield", dims=dims[("e",)], values=np.array([0.92, 0.1, 0.92])
        ),
        "demolition yield": Parameter(
            name="demolition yield", dims=dims[("e",)], values=np.array([0.95, 0.02, 0.95])
        ),
        "remelting yield": Parameter(
            name="remelting yield", dims=dims[("e",)], values=np.array([0.96, 1.0, 0.5])
        ),
    }

    process_names = [
        "sysenv",
        "shredder",
        "demolition",
        "remelting",
        "landfills",
        "slag piles",
    ]
    processes = make_processes(process_names)

    flow_definitions = [
        FlowDefinition(
            from_process_name="sysenv", to_process_name="shredder", dim_letters=("t", "e")
        ),
        FlowDefinition(
            from_process_name="sysenv", to_process_name="demolition", dim_letters=("t", "e")
        ),
        FlowDefinition(
            from_process_name="shredder",
            to_process_name="remelting",
            dim_letters=("t", "e"),
        ),  # scrap type 1
        FlowDefinition(
            from_process_name="shredder", to_process_name="sysenv", dim_letters=("t", "e")
        ),  # shredder residue
        FlowDefinition(
            from_process_name="demolition",
            to_process_name="remelting",
            dim_letters=("t", "e"),
        ),  # scrap type 2
        FlowDefinition(
            from_process_name="demolition",
            to_process_name="landfills",
            dim_letters=("t", "e"),
        ),  # loss
        FlowDefinition(
            from_process_name="remelting",
            to_process_name="slag piles",
            dim_letters=("t", "e"),
        ),  # secondary steel
        FlowDefinition(
            from_process_name="remelting", to_process_name="sysenv", dim_letters=("t", "e")
        ),  # slag
    ]
    flows = make_empty_flows(processes, flow_definitions, dims)

    stock_definitions = [
        StockDefinition(
            name="landfills",
            process="landfills",
            dim_letters=("t", "e"),
            subclass=SimpleFlowDrivenStock,
        ),
        StockDefinition(
            name="slag piles",
            process="slag piles",
            dim_letters=("t", "e"),
            subclass=SimpleFlowDrivenStock,
        ),
    ]
    stocks = make_empty_stocks(stock_definitions, processes, dims)

    return ExampleMFA(
        dims=dims, parameters=parameters, processes=processes, flows=flows, stocks=stocks
    )
