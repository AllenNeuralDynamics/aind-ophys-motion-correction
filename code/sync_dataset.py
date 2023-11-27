import datetime
import warnings
from functools import lru_cache

import h5py as h5
import numpy as np

warnings.simplefilter("ignore")  # ignore h5py warnings


@lru_cache(maxsize=32)
def get_bit(uint_array, bit):
    """
    Returns a bool array for a specific bit in a uint ndarray.

    Parameters
    ----------
    uint_array : (numpy.ndarray)
        The array to extract bits from.
    bit : (int)
        The bit to extract.

    """
    return np.bitwise_and(uint_array, 2**bit).astype(bool).astype(np.uint8)


class Sync(object):
    """
    Sync Data Object
    """

    def __init__(self, path):
        self._data = h5.File(path, "r")
        self._meta_data = {}

    @property
    def meta_data(self):
        if not self._meta_data:
            self._meta_data = eval(self._data["meta"][()])
            self.meta_data["start_time"] = datetime.datetime.fromisoformat(self.meta_data["start_time"])
            self.meta_data["stop_time"] = datetime.datetime.fromisoformat(self.meta_data["stop_time"])
            self.meta_data["duration_s"] = (self.meta_data["stop_time"] - self.meta_data["start_time"]).total_seconds()
        return self._meta_data

    @property
    def line_labels(self):
        return self.meta_data["line_labels"]

    @property
    def timestamps(self):
        times = self.get_all_events()[:, 0:1].astype(np.int64)
        intervals = np.ediff1d(times, to_begin=0)
        rollovers = np.where(intervals < 0)[0]
        for i in rollovers:
            times[i:] += 4294967296
        return times

    @property
    def sample_freq(self):
        try:
            return float(self.meta_data["ni_daq"]["sample_freq"])
        except KeyError:
            return float(self.meta_data["ni_daq"]["counter_output_freq"])

    @lru_cache(maxsize=32)
    def get_bit(self, bit):
        return self.bit(bit)

    @lru_cache(maxsize=32)
    def get_first_bit(self, line):
        value = self._data["data"][0][0]
        return value & 2**line

    def bit(self, bit):
        return np.bitwise_and(self.get_all_bits(), 2**bit).astype(bool).astype(np.uint8)

    def line(self, line):
        """
        Returns the values for a specific line.

        Parameters
        ----------
        line : str
            Line to return.

        """
        bit = self._line_to_bit(line)
        return self.get_bit(bit)

    def get_bit_changes(self, bit):
        """
        Returns the first derivative of a specific bit.
            Data points are 1 on rising edges and 255 on falling edges.

        Parameters
        ----------
        bit : int
            Bit for which to return changes.

        """
        bit_array = self.get_bit(bit)
        return np.ediff1d(bit_array, to_begin=0)

    def get_line_changes(self, line):
        """
        Returns the first derivative of a specific line.
            Data points are 1 on rising edges and 255 on falling edges.

        Parametersmmm
        ----------
        line : (str)
            Line name for which to return changes.

        """
        bit = self._line_to_bit(line)
        return self.get_bit_changes(bit)

    def get_all_bits(self):
        """
        Returns the data for all bits.

        """
        return self._data["data"][()][:, -1]

    def get_all_times(self, units="samples"):
        """
        Returns all counter values.

        Parameters
        ----------
        units : str
            Return times in 'samples' or 'seconds'

        """
        if self.meta_data["ni_daq"]["counter_bits"] == 32:
            times = self.get_all_events()[:, 0]
        else:
            times = self.times
        units = units.lower()
        if units == "samples":
            return times
        elif units in ["seconds", "sec", "secs"]:
            freq = self.sample_freq
            return times / freq
        else:
            raise ValueError("Only 'samples' or 'seconds' are valid units.")

    # @lru_cache(maxsize=32)
    def get_all_events(self, units="samples"):
        """
        Returns all counter values and their cooresponding IO state.
        """
        if units == "samples":
            return self._data["data"][()]
        else:
            return self._data["data"][()] / self.sample_freq

    @lru_cache(maxsize=32)
    def get_events_by_bit(self, bit, units="samples"):
        """
        Returns all counter values for transitions (both rising and falling)
            for a specific bit.

        Parameters
        ----------
        bit : int
            Bit for which to return events.

        """
        changes = self.get_bit_changes(bit)
        return self.get_all_times(units)[np.where(changes != 0)]

    @lru_cache(maxsize=32)
    def get_events_by_line(self, line, units="samples"):
        """
        Returns all counter values for transitions (both rising and falling)
            for a specific line.

        Parameters
        ----------
        line : str
            Line for which to return events.

        """
        line = self._line_to_bit(line)
        return self.get_events_by_bit(line, units)

    def _line_to_bit(self, line):
        """
        Returns the bit for a specified line.  Either line name and number is
            accepted.

        Parameters
        ----------
        line : str
            Line name for which to return corresponding bit.

        """
        if type(line) is int:
            return line
        elif type(line) is str:
            return self.line_labels.index(line)
        else:
            raise TypeError("Incorrect line type.  Try a str or int.")

    def _bit_to_line(self, bit):
        """
        Returns the line name for a specified bit.

        Parameters
        ----------
        bit : int
            Bit for which to return the corresponding line name.
        """
        return self.line_labels[bit]

    def get_rising_edges(self, line, units="samples"):
        """
        Returns the counter values for the rizing edges for a specific bit or
            line.

        Parameters
        ----------
        line : str
            Line for which to return edges.

        """
        bit = self._line_to_bit(line)
        changes = self.get_bit_changes(bit)
        return self.get_all_times(units)[np.where(changes == 1)]

    def get_falling_edges(self, line, units="samples"):
        """
        Returns the counter values for the falling edges for a specific bit
            or line.

        Parameters
        ----------
        line : str
            Line for which to return edges.

        """
        bit = self._line_to_bit(line)
        changes = self.get_bit_changes(bit)
        return self.get_all_times(units)[np.where(changes == 255)]

    def get_nearest(
        self,
        source,
        target,
        source_edge="rising",
        target_edge="rising",
        direction="previous",
        units="indices",
    ):
        """
        For all values of the source line, finds the nearest edge from the
            target line.

        By default, returns the indices of the target edges.

        Args:
            source (str, int): desired source line
            target (str, int): desired target line
            source_edge [Optional(str)]: "rising" or "falling" source edges
            target_edge [Optional(str): "rising" or "falling" target edges
            direction (str): "previous" or "next". Whether to prefer the
                previous edge or the following edge.
            units (str): "indices"

        """
        source_edges = getattr(self, "get_{}_edges".format(source_edge.lower()))(source.lower(), units="samples")
        target_edges = getattr(self, "get_{}_edges".format(target_edge.lower()))(target.lower(), units="samples")
        indices = np.searchsorted(target_edges, source_edges, side="right")
        if direction.lower() == "previous":
            indices[np.where(indices != 0)] -= 1
        elif direction.lower() == "next":
            indices[np.where(indices == len(target_edges))] = -1
        if units in ["indices", "index"]:
            return indices
        elif units == "samples":
            return target_edges[indices]
        elif units in ["sec", "seconds", "second"]:
            return target_edges[indices] / self.sample_freq
        else:
            raise KeyError("Invalid units.  Try 'seconds', 'samples' or 'indices'")

    @lru_cache(maxsize=32)
    def line_stats(self, line, print_results=False):
        """
        Quick-and-dirty analysis of a bit.

        ##TODO: Split this up into smaller functions.

        """
        # convert to bit
        bit = self._line_to_bit(line)

        # get the bit's data
        bit_data = self.get_bit(bit)
        total_data_points = len(bit_data)

        # get the events
        events = self.get_events_by_bit(bit)
        total_events = len(events)

        # get the rising edges
        rising = self.get_rising_edges(bit)
        total_rising = len(rising)

        # get falling edges
        falling = self.get_falling_edges(bit)
        total_falling = len(falling)

        if total_events <= 0:
            if print_results:
                print("*" * 70)
                print("No events on line: %s" % line)
                print("*" * 70)
            return {
                "line": line,
                "bit": bit,
            }
        elif total_events <= 10:
            if print_results:
                print("*" * 70)
                print("Sparse events on line: %s" % line)
                print("Rising: %s" % total_rising)
                print("Falling: %s" % total_falling)
                print("*" * 70)
            return {
                "line": line,
                "total_rising": total_rising,
                "total_falling": total_falling,
                "avg_freq": None,
                ##"duty_cycle": None,
            }
        else:
            # period
            period = self.period(line)

            avg_period = period["avg"]
            max_period = period["max"]
            min_period = period["min"]
            period_sd = period["sd"]

            # freq
            avg_freq = self.frequency(line)

            # duty cycle
            duty_cycle = self.duty_cycle(line)

            if print_results:
                print("*" * 70)

                print("Quick stats for line: %s" % line)
                print("Bit: %i" % bit)
                print("Data points: %i" % total_data_points)
                print("Total transitions: %i" % total_events)
                print("Rising edges: %i" % total_rising)
                print("Falling edges: %i" % total_falling)
                print("Average period: %s" % avg_period)
                print("Minimum period: %s" % min_period)
                print("Max period: %s" % max_period)
                print("Period SD: %s" % period_sd)
                print("Average freq: %s" % avg_freq)
                print("Duty cycle: %s" % duty_cycle)

                print("*" * 70)

            return {
                "line": line,
                "total_data_points": total_data_points,
                "total_events": total_events,
                "total_rising": total_rising,
                "total_falling": total_falling,
                "avg_period": round(avg_period, 2),
                "min_period": round(min_period, 2),
                "max_period": round(max_period, 2),
                "period_sd": round(period_sd, 2),
                "avg_freq": round(avg_freq, 2),
                "duty_cycle": duty_cycle,
            }

    def period(self, line, edge="rising"):
        """
        Returns a dictionary with avg, min, max, and st of period for a line.
        """
        bit = self._line_to_bit(line)

        if edge.lower() == "rising":
            edges = self.get_rising_edges(bit)
        elif edge.lower() == "falling":
            edges = self.get_falling_edges(bit)

        if len(edges) > 2:
            timebase_freq = self.meta_data["ni_daq"]["counter_output_freq"]
            avg_period = np.mean(np.ediff1d(edges[1:])) / timebase_freq
            max_period = np.max(np.ediff1d(edges[1:])) / timebase_freq
            min_period = np.min(np.ediff1d(edges[1:])) / timebase_freq
            period_sd = np.std(avg_period)

        else:
            raise IndexError("Not enough edges for period: %i" % len(edges))

        return {
            "avg": avg_period,
            "max": max_period,
            "min": min_period,
            "sd": period_sd,
        }

    def frequency(self, line, edge="rising"):
        """
        Returns the average frequency of a line.
        """

        period = self.period(line, edge)
        return 1.0 / period["avg"]

    def duty_cycle(self, line):
        """
        Doesn't work right now.  Freezes python for some reason.

        Returns the duty cycle of a line.

        """
        bit = self._line_to_bit(line)

        rising = self.get_rising_edges(bit)
        falling = self.get_falling_edges(bit)

        total_rising = len(rising)
        total_falling = len(falling)

        if total_rising > total_falling:
            rising = rising[:total_falling]
        elif total_rising < total_falling:
            falling = falling[:total_rising]
        else:
            pass

        if rising[0] < falling[0]:
            # line starts low
            high = falling - rising
        else:
            # line starts high
            high = np.concatenate(falling, self.get_all_events()[-1, 0]) - np.concatenate(0, rising)

        total_high_time = np.sum(high)
        all_events = self.get_events_by_bit(bit)
        total_time = all_events[-1] - all_events[0]
        return 1.0 * total_high_time / total_time

    def stats(self):
        """
        Quick-and-dirty analysis of all bits.  Prints a few things about each
            bit where events are found.
        """
        bits = []
        for i in range(32):
            bits.append(self.line_stats(i, print_results=False))
        active_bits = [x for x in bits if x is not None]
        print("Active bits: ", len(active_bits))
        for bit in active_bits:
            print("*" * 70)
            print("Bit: %i" % bit["bit"])
            print("Label: %s" % self.line_labels[bit["bit"]])
            print("Rising edges: %i" % bit["total_rising"])
            print("Falling edges: %i" % bit["total_falling"])
            print("Average freq: %s" % bit["avg_freq"])
            print("Duty cycle: %s" % bit["duty_cycle"])
        print("*" * 70)
        return active_bits

    def close(self):
        """
        Closes the dataset.
        """
        self._data.close()

    def __enter__(self):
        """
        So we can use context manager (with...as) like any other open file.

        Examples
        --------
        >>> with Dataset('my_data.h5') as d:
        ...     d.stats()

        """
        return self

    def __exit__(self, type, value, traceback):
        """
        Exit statement for context manager.
        """
        self.close()