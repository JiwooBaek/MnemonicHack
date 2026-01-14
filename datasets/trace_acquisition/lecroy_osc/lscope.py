import struct
import numpy as np
import socket
import sys
import datetime
import traceback
from typing import Optional, Tuple
from .format import *


class LScope:
    def __init__(self,
                 num_of_channels: int = 4
                 ):
        self.sock: Optional[socket] = None
        self.setting_commands = ['GRID', 'TIME_DIV', 'COMM_FORMAT', 'COMM_HEADER', 'COMM_ORDER'] + \
                                ['TRIG_DELAY', 'TRIG_SELECT', 'TRIG_MODE', 'TRIG_PATTERN', 'SEQUENCE', 'BWL'] + \
                                ['C%i:COUPLING' % i for i in range(1, num_of_channels + 1)] + \
                                ['C%i:VOLT_DIV' % i for i in range(1, num_of_channels + 1)] + \
                                ['C%i:OFFSET' % i for i in range(1, num_of_channels + 1)] + \
                                ['C%i:TRIG_COUPLING' % i for i in range(1, num_of_channels + 1)] + \
                                ['C%i:TRIG_LEVEL' % i for i in range(1, num_of_channels + 1)] + \
                                ['C%i:TRIG_SLOPE' % i for i in range(1, num_of_channels + 1)] + \
                                ['C%i:TRACE' % i for i in range(1, num_of_channels + 1)]
        self._header_fmt = '>BBBBL'
        self._num_of_channels = num_of_channels
        pass

    def __del__(self):
        self.disconnect()

    def connect(self, host, port=1861, timeout=5.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock.settimeout(timeout)
        self.clear_recv_buf()
        self.send('comm_header short')
        self.check_last_command()
        self.send('CFMT DEF9,WORD,BIN')
        self.check_last_command()
        self.send('seq off')
        print(f"Lecroy scope({host}:{port}) is connected!\n(default waveform dtype: signed 16-byte int.)")
        pass

    def disconnect(self):
        self.sock.close()
        pass

    def clear_recv_buf(self, timeout=0.5):
        """
        Clear any bytes in the oscilloscope's output queue by receiving
        packets until the connection blocks for more than `timeout` seconds.
        """
        t = self.sock.gettimeout()
        self.sock.settimeout(timeout)
        try:
            while True:
                self.sock.recv(4096)
        except socket.timeout:
            pass
        self.sock.settimeout(t)
        pass

    def send(self, msg) -> None:
        """
        Format and send the string `msg`.
        """
        if not msg.endswith('\n'):
            msg += '\n'
        # operation, headerver, seqnum, spare, totalbytes
        header = struct.pack(self._header_fmt, 129, 1, 1, 0, len(msg))

        self.sock.sendall(header + msg.encode('ascii'))
        pass

    def recv(self) -> bytes:
        """
        Return a message from the scope.
        """
        reply = ''.encode('ascii')
        while True:
            header = ''.encode('ascii')
            while len(header) < 8:
                header += self.sock.recv(8 - len(header))
            operation, headerver, seqnum, spare, totalbytes = \
                struct.unpack(self._header_fmt, header)
            buffer = ''.encode('ascii')
            while len(buffer) < totalbytes:
                buffer += self.sock.recv(totalbytes - len(buffer))
            reply += buffer
            if operation % 2:
                break
        return reply

    def check_last_command(self) -> None:
        """
        Check that the last command sent was received okay; if not, raise
        an exception with details about the error.
        """
        self.send('cmr?')
        err = int(self.recv().split()[-1].rstrip('\n'.encode('ascii')))

        if err in errors:
            self.sock.close()
            raise Exception(errors[err])
        pass

    def query(self,
              query_str: str,
              verbose: bool = False
              ) -> Optional[str]:
        if "?" not in query_str.strip():
            query_str = query_str.strip() + "?"
        if verbose:
            print("QUERY: " + query_str)
        self.send(query_str)
        try:
            result = self.recv().strip().decode('ascii')
            self.check_last_command()
        except socket.timeout:
            print(f"[LeCroy-OSC] Time out.", file=sys.stderr)
            return None
        except Exception as e:
            print(f"{e.args[0]} : {query_str}", file=sys.stderr)
            return None
        return result

    def get_settings(self) -> dict:
        """
        Captures the current settings of the scope as a dict of Command->Setting.
        """
        settings = {}
        for command in self.setting_commands:
            self.send(command + '?')
            settings[command] = self.recv().strip().decode('ascii')
            self.check_last_command()
        return settings

    def set_settings(self,
                     settings: dict,
                     verbose: bool = False
                     ) -> bool:
        """
        Sends a `settings` dict of Command->Setting to the scope.
        """
        setting = ''
        try:
            for command, setting in settings.items():
                if verbose:
                    print('sending %s' % command)
                self.send(setting)
                self.check_last_command()
        except socket.timeout:
            print("[LeCroy-OSC] Time out.", file=sys.stderr)
            return False
        except Exception as e:
            print(f"{e.args[0]} : {setting}", file=sys.stderr)
            return False
        return True

    def get_active_channels(self) -> list:
        """
        Returns a list of the active channels on the scope.
        """
        channels = []
        for i in range(1, self._num_of_channels + 1):
            self.send('c%i:trace?' % i)
            if 'ON'.encode('ascii') in self.recv():
                channels.append(i)
        return channels

    def get_current_waveform_length(self) -> int:
        self.clear_recv_buf()
        self.send("TRMD SINGLE;ARM;FRTR")
        w = self.get_waveform(1, seq_mode=False)
        return w[1].shape[0]

    def set_waveform_dtype(self, dtype: str):
        assert dtype.upper() in ('BYTE', 'WORD')
        self.send(f'CFMT DEF9,{dtype.upper()},BIN')
        self.check_last_command()
        pass

    def arm(self, wait_time: Optional[int] = 10) -> None:
        """
        Arms the oscilliscope and instructs it to wait before processing
        further commands, i.e. nonblocking.
        """
        if wait_time is None:
            self.send('arm')
        else:
            assert type(wait_time) is int
            self.send(f'arm;wait {wait_time}')
        pass

    def stop(self):
        self.send('stop')
        pass

    def check_is_stop(self):
        try:
            resp = self.query("TRIG_MODE?")
            if 'STOP' in resp.upper():
                return True
            else:
                return False
        except Exception:
            print("[LeCroy-OSC] Trigger mode check failed.", file=sys.stderr)
            return False

    def check_is_stop_seq_mode(self):
        resp = self.query("TRIG_MODE?")
        try:
            if 'STOP' in resp.upper():
                return True
            else:
                return False
        except Exception:
            return False

    def set_sequence_mode(self, nsequence: int) -> None:
        """
        Sets the scope to use sequence mode for aquisition.
        """
        assert (type(nsequence) is int) and 1 <= nsequence
        if nsequence == 1:
            self.send('seq off')
        else:
            self.send('seq on,%i' % nsequence)
        pass

    def _get_wavedesc(self, channel: int) -> dict:
        """
        Requests the wave descriptor for `channel` from the scope. Returns it in
        dictionary format.
        """
        assert 1 <= channel <= self._num_of_channels

        self.send('c%s:wf? desc' % str(channel))

        msg = self.recv()
        if not int(msg[0:6].decode()[1]) == channel:
            raise RuntimeError('waveforms out of sync or comm_header is off.')

        # data = io.StringIO(msg.decode('ascii', 'replace'))
        # startpos = re.search('WAVEDESC', data.read()).start()
        data = msg
        startpos = msg.find(b'WAVEDESC')

        wavedesc = {}

        # check endian
        # data.seek(startpos + 34)
        if struct.unpack('<' + Enum.fmt_str, data[startpos + 34:startpos + 34 + Enum.length]) == 0:
            endian = '>'
            wavedesc['little_endian'] = True
            # np.little_endian = True
        else:
            endian = '<'
            wavedesc['little_endian'] = False
            # raise RuntimeError('Not supported: Big_endian.')
            # np.little_endian = False
        # data.seek(startpos)

        # build dictionary of wave description
        data = msg[startpos:]
        for name, pos, datatype in wavedesc_template:
            # raw = data.read(datatype.length)
            raw = data[:datatype.length]
            data = data[datatype.length:]
            if datatype in (String, UnitDefinition):
                wavedesc[name] = raw.rstrip(b'\x00')
            elif datatype in (TimeStamp,):
                wavedesc[name] = struct.unpack(endian + datatype.fmt_str, raw)
            else:
                wavedesc[name] = struct.unpack(endian + datatype.fmt_str, raw)[0]

        # determine data type
        if wavedesc['comm_type'] == 0:
            wavedesc['dtype'] = np.int8()
        elif wavedesc['comm_type'] == 1:
            wavedesc['dtype'] = np.int16()
        else:
            raise Exception('unknown comm_type.')
        return wavedesc

    def get_waveform_v2(self,
                        channel: int,
                        seq_mode: bool = False,
                        timeout: float = 0.5,
                        print_err_msg: bool = False
                        ) -> Optional[Tuple[dict, np.ndarray]]:
        assert 0 < timeout < 10
        t = self.sock.gettimeout()
        self.sock.settimeout(timeout)
        self.send(f"c{channel}:wf? dat1")
        buf = []
        while True:
            try:
                _t = self.sock.recv(4096)
                buf.append(_t)
            except Exception:
                break
            pass
        self.sock.settimeout(t)
        try:
            buf = b''.join(buf)
            nbytes_buf = len(buf)

            trc_buf = []
            cur_pos = 0
            while cur_pos < nbytes_buf:
                _, _, _, _, seg_bytes = struct.unpack(self._header_fmt, buf[cur_pos: cur_pos + 8])
                cur_pos += 8
                trc_buf.append(buf[cur_pos: cur_pos + seg_bytes])
                cur_pos += seg_bytes
                pass
            trc_buf = b''.join(trc_buf)

            wavedesc = self._get_wavedesc(channel)
            trace = np.frombuffer(trc_buf[22:], wavedesc['dtype'], wavedesc['wave_array_count'])

            if seq_mode:
                seq_info = self.query("seq?").split()[1].split(",")
                assert seq_info[0].lower() == 'on'
                seq_num = int(seq_info[1])
                return wavedesc, trace.reshape((seq_num, trace.shape[0] // seq_num))
            else:
                return wavedesc, trace
        except Exception as e:
            if print_err_msg:
                print(traceback.format_exc(), file=sys.stderr)
            return None
        pass

    def get_waveform(self,
                     channel: int,
                     seq_mode: bool = False,
                     pre_clear_buffer: bool = False,
                     ) -> Optional[Tuple[dict, np.ndarray]]:
        """
        Capture the raw data for `channel` from the scope and return a tuple
        containing the wave descriptor and a numpy array of the digitized
        scope readout.
        """
        assert 1 <= channel <= self._num_of_channels
        if pre_clear_buffer:
            self.clear_recv_buf(timeout=0.1)
        self.send('c%s:wf? dat1' % str(channel))
        try:
            msg = self.recv()
            assert int(msg[0:6].decode()[1]) == channel, 'waveforms out of sync or comm_header is off.'
            wavedesc = self._get_wavedesc(channel)
        except socket.timeout:
            return None

        if seq_mode:
            seq_info = self.query("seq?").split()[1].split(",")
            assert seq_info[0].lower() == 'on'
            seq_num = int(seq_info[1])
            raw_trc = np.fromstring(msg[22:], wavedesc['dtype'], wavedesc['wave_array_count'])
            return wavedesc, raw_trc.reshape((seq_num, raw_trc.shape[0] // seq_num))
        else:
            return wavedesc, np.fromstring(msg[22:], wavedesc['dtype'], wavedesc['wave_array_count'])
        pass

    def get_screen_image(self,
                         img_id: str,
                         bg_color: str = "BLACK",
                         area: str = "DSOWINDOW",
                         store_path: str = "./"
                         ) -> None:
        assert bg_color.upper() in ('BLACK', 'WHITE')
        assert area.upper() in ('DSOWINDOW', 'GRIDAREAONLY')
        self.clear_recv_buf()
        self.send(f"HCSU BCKG,{bg_color.upper()};HCSU DEV,PNG;HCSU PORT,GPIB;HCSU AREA,{area.upper()}")
        self.check_last_command()
        self.send(f"SCDP")
        raw_png = self.recv()
        img_id = f"{img_id.replace('.png', '')}-lecroy-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        file_stream = open(f"{store_path}/{img_id}", 'wb')
        file_stream.write(raw_png)
        file_stream.close()
        pass
    pass
