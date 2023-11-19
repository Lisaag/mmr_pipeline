import _plotly_utils.basevalidators


class TicklabeloverflowValidator(_plotly_utils.basevalidators.EnumeratedValidator):
    def __init__(
        self,
        plotly_name="ticklabeloverflow",
        parent_name="densitymapbox.colorbar",
        **kwargs,
    ):
        super(TicklabeloverflowValidator, self).__init__(
            plotly_name=plotly_name,
            parent_name=parent_name,
            edit_type=kwargs.pop("edit_type", "colorbars"),
            values=kwargs.pop("values", ["allow", "hide past div", "hide past domain"]),
            **kwargs,
        )
