import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from threading import Thread

try:
    import pyds
    PYDS_AVAILABLE = True
except ImportError:
    PYDS_AVAILABLE = False
    import warnings
    warnings.warn(
        "pyds not available. Multi-stream mode disabled. "
        "Only single-camera mode will work."
    )

from camera.schemas import RTSPConfig

logger = logging.getLogger(__name__)


class MultiStreamManager:
    """
    Efficient multi-stream manager for handling 100+ cameras
    Uses DeepStream batched processing for optimal GPU utilization
    """
    
    def __init__(self, batch_size: int = 30):
        self.batch_size = batch_size
        self.cameras: Dict[str, Dict] = {}
        
        # Pipeline components
        self.pipeline = None
        self.loop = None
        self.thread = None
        self.is_running = False
        
        # Performance metrics
        self.total_frames = 0
        self.start_time = None
        
    def _create_source_bin(self, index: int, rtsp_url: str) -> Gst.Bin:
        """Create source bin for each RTSP stream with error recovery"""
        bin_name = f"source-bin-{index:03d}"
        nbin = Gst.Bin.new(bin_name)
        
        # RTSP source with optimized settings
        source = Gst.ElementFactory.make("rtspsrc", f"source-{index}")
        source.set_property("location", rtsp_url)
        source.set_property("latency", 100)
        source.set_property("drop-on-latency", True)
        source.set_property("protocols", "tcp")  # Force TCP for reliability
        source.set_property("retry", 5)
        source.set_property("timeout", 5000000)  # 5 seconds
        
        # Depayloader and parser
        depay = Gst.ElementFactory.make("rtph264depay", f"depay-{index}")
        parser = Gst.ElementFactory.make("h264parse", f"parser-{index}")
        
        # Hardware decoder
        decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder-{index}")
        decoder.set_property("enable-max-performance", 1)
        decoder.set_property("drop-frame-interval", 0)
        decoder.set_property("num-extra-surfaces", 1)
        
        # Add elements to bin
        nbin.add(source)
        nbin.add(depay)
        nbin.add(parser)
        nbin.add(decoder)
        
        # Link static elements
        depay.link(parser)
        parser.link(decoder)
        
        # Connect pad-added signal for dynamic RTSP linking
        source.connect("pad-added", self._on_pad_added, depay)
        
        # Create ghost pad
        decoder_src_pad = decoder.get_static_pad("src")
        if not decoder_src_pad:
            logger.error(f"Failed to get decoder src pad for source {index}")
            return None
        
        bin_pad = Gst.GhostPad.new("src", decoder_src_pad)
        nbin.add_pad(bin_pad)
        
        logger.debug(f"Created source bin {index} for {rtsp_url}")
        return nbin
    
    def _on_pad_added(self, element, pad, dest_element):
        """Callback for dynamically linking RTSP pads"""
        sink_pad = dest_element.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)
    
    def _build_multi_stream_pipeline(self, streams: List[RTSPConfig], width: int, height: int) -> bool:
        """Build batched pipeline for multiple streams"""
        num_sources = len(streams)
        logger.info(f"Building multi-stream pipeline for {num_sources} cameras")
        
        try:
            # Create pipeline
            self.pipeline = Gst.Pipeline.new("multi-stream-pipeline")
            
            # Create streammux for batching
            streammux = Gst.ElementFactory.make("nvstreammux", "mux")
            streammux.set_property("batch-size", num_sources)
            streammux.set_property("width", width)
            streammux.set_property("height", height)
            streammux.set_property("batched-push-timeout", 40000)
            streammux.set_property("live-source", 1)
            streammux.set_property("enable-padding", 0)
            streammux.set_property("nvbuf-memory-type", 0)  # Use default memory
            self.pipeline.add(streammux)
            
            # Add source bins
            for idx, stream in enumerate(streams):
                source_bin = self._create_source_bin(idx, stream.rtsp_url)
                if not source_bin:
                    raise Exception(f"Failed to create source bin for stream {idx}")
                
                self.pipeline.add(source_bin)
                
                # Link to streammux
                srcpad = source_bin.get_static_pad("src")
                sinkpad = streammux.get_request_pad(f"sink_{idx}")
                if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                    raise Exception(f"Failed to link source {idx} to mux")
                
                # Store camera metadata
                self.cameras[stream.camera_id] = {
                    'index': idx,
                    'config': stream,
                    'current_frame': None,
                    'frame_count': 0,
                    'roi_mask': None,
                    'last_frame_time': 0
                }
                
                logger.info(f"Added camera {stream.camera_id} at index {idx}")
            
            # Create tiler for batched visualization
            tiler = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
            rows = int(np.ceil(np.sqrt(num_sources)))
            cols = int(np.ceil(num_sources / rows))
            tiler.set_property("rows", rows)
            tiler.set_property("columns", cols)
            tiler.set_property("width", 1920)
            tiler.set_property("height", 1080)
            self.pipeline.add(tiler)
            
            # Video converter
            nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
            self.pipeline.add(nvvidconv)
            
            # Caps filter for format
            capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
            caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
            capsfilter.set_property("caps", caps)
            self.pipeline.add(capsfilter)
            
            # OSD for metadata overlay (optional)
            nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
            self.pipeline.add(nvosd)
            
            # Final converter
            nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
            self.pipeline.add(nvvidconv2)
            
            capsfilter2 = Gst.ElementFactory.make("capsfilter", "filter2")
            caps2 = Gst.Caps.from_string("video/x-raw, format=BGRx")
            capsfilter2.set_property("caps", caps2)
            self.pipeline.add(capsfilter2)
            
            # Appsink for frame extraction
            appsink = Gst.ElementFactory.make("appsink", "appsink")
            appsink.set_property("emit-signals", True)
            appsink.set_property("max-buffers", 1)
            appsink.set_property("drop", True)
            appsink.set_property("sync", False)
            appsink.connect("new-sample", self._on_new_sample)
            self.pipeline.add(appsink)
            
            # Link all elements
            if not streammux.link(tiler):
                raise Exception("Failed to link streammux to tiler")
            if not tiler.link(nvvidconv):
                raise Exception("Failed to link tiler to nvvidconv")
            if not nvvidconv.link(capsfilter):
                raise Exception("Failed to link nvvidconv to capsfilter")
            if not capsfilter.link(nvosd):
                raise Exception("Failed to link capsfilter to nvosd")
            if not nvosd.link(nvvidconv2):
                raise Exception("Failed to link nvosd to nvvidconv2")
            if not nvvidconv2.link(capsfilter2):
                raise Exception("Failed to link nvvidconv2 to capsfilter2")
            if not capsfilter2.link(appsink):
                raise Exception("Failed to link capsfilter2 to appsink")
            
            # Add bus watch
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)
            
            logger.info(f"Multi-stream pipeline built successfully with {num_sources} cameras")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build multi-stream pipeline: {str(e)}")
            return False
    
    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages"""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"Pipeline error: {err}, Debug: {debug}")
            
        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(f"Pipeline warning: {warn}")
            
        elif msg_type == Gst.MessageType.EOS:
            logger.info("Pipeline end of stream")
            self.is_running = False
    
    def _on_new_sample(self, appsink):
        """Process batched frames with metadata"""
        try:
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR
            
            buffer = sample.get_buffer()
            
            # Get batch metadata using pyds
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            
            l_frame = batch_meta.frame_meta_list
            while l_frame is not None:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                    source_id = frame_meta.source_id
                    
                    # Find camera by source_id
                    camera_id = None
                    for cam_id, cam_data in self.cameras.items():
                        if cam_data['index'] == source_id:
                            camera_id = cam_id
                            break
                    
                    if camera_id:
                        # Update frame metadata
                        self.cameras[camera_id]['frame_count'] += 1
                        self.cameras[camera_id]['last_frame_time'] = datetime.now().timestamp()
                        self.total_frames += 1
                    
                except StopIteration:
                    break
                
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
                    
        except Exception as e:
            logger.error(f"Error processing batched sample: {str(e)}")
            return Gst.FlowReturn.ERROR
        
        return Gst.FlowReturn.OK
    
    def _run_loop(self):
        """Run GLib main loop"""
        try:
            self.loop = GLib.MainLoop()
            logger.info("Starting multi-stream GLib main loop")
            self.loop.run()
        except Exception as e:
            logger.error(f"Error in multi-stream GLib loop: {str(e)}")
        finally:
            logger.info("Multi-stream GLib main loop stopped")
    
    async def start(self, streams: List[RTSPConfig], width: int, height: int) -> bool:
        """Start multi-stream pipeline"""
        try:
            if self.is_running:
                logger.warning("Multi-stream manager already running")
                return False
            
            # Build pipeline
            if not self._build_multi_stream_pipeline(streams, width, height):
                raise Exception("Failed to build multi-stream pipeline")
            
            # Start pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise Exception("Failed to start multi-stream pipeline")
            
            # Start GLib loop
            self.is_running = True
            self.start_time = datetime.now()
            self.thread = Thread(target=self._run_loop, daemon=True, name="MultiStream-GLib")
            self.thread.start()
            
            logger.info(f"Multi-stream manager started with {len(streams)} cameras")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start multi-stream manager: {str(e)}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop multi-stream pipeline"""
        try:
            logger.info("Stopping multi-stream manager...")
            self.is_running = False
            
            # Stop pipeline
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            
            # Stop GLib loop
            if self.loop and self.loop.is_running():
                self.loop.quit()
            
            # Wait for thread
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    logger.warning("Multi-stream thread did not stop gracefully")
            
            # Cleanup
            self.cameras.clear()
            self.pipeline = None
            self.loop = None
            self.thread = None
            
            logger.info("Multi-stream manager stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping multi-stream manager: {str(e)}")
    
    def get_camera_status(self, camera_id: str) -> Optional[Dict]:
        """Get status for specific camera"""
        if camera_id not in self.cameras:
            return None
        
        cam_data = self.cameras[camera_id]
        return {
            'camera_id': camera_id,
            'is_running': self.is_running,
            'frame_count': cam_data['frame_count'],
            'last_frame_time': cam_data['last_frame_time'],
            'stream_index': cam_data['index'],
            'backend': 'deepstream-multi'
        }
    
    def get_overall_status(self) -> Dict:
        """Get overall pipeline status"""
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'is_running': self.is_running,
            'num_cameras': len(self.cameras),
            'camera_ids': list(self.cameras.keys()),
            'total_frames': self.total_frames,
            'uptime_seconds': uptime,
            'backend': 'deepstream-multi'
        }