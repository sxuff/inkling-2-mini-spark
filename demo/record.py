"""Real-time capture of index.html streaming from the live service, via CDP screencast -> H.264 MP4.
usage: python record.py OUT.mp4 [page-query-string]"""
import asyncio, base64, http.server, json, os, shutil, subprocess, sys, threading, time
from functools import partial
from pathlib import Path
from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve()
QUERY = sys.argv[2] if len(sys.argv) > 2 else ""
FRAMES = HERE / "frames"
HOLD_END = 3.0


UPSTREAM = ("127.0.0.1", int(os.environ.get("UPSTREAM_PORT", "8020")))


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static page + same-origin streaming proxy of /v1/* to the tunnelled API."""

    def log_message(self, *a, **k):
        pass

    def do_POST(self):
        import http.client
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        up = http.client.HTTPConnection(*UPSTREAM, timeout=600)
        up.request("POST", self.path, body, {"Content-Type": "application/json"})
        resp = up.getresponse()
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "text/event-stream"))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        while chunk := resp.read1(65536):
            self.wfile.write(chunk)
            self.wfile.flush()
        up.close()


def serve():
    h = partial(Handler, directory=str(HERE))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


async def main():
    srv = serve()
    port = srv.server_address[1]
    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    FRAMES.mkdir()
    frames = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        await page.goto(f"http://127.0.0.1:{port}/index.html?{QUERY}")
        await page.wait_for_timeout(1500)
        cdp = await page.context.new_cdp_session(page)

        async def on_frame(ev):
            ts = ev["metadata"]["timestamp"]
            path = FRAMES / f"{len(frames):06d}.jpg"
            path.write_bytes(base64.b64decode(ev["data"]))
            frames.append((ts, path))
            try:
                await cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})
            except Exception:
                pass

        cdp.on("Page.screencastFrame", lambda ev: asyncio.ensure_future(on_frame(ev)))
        await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 95, "maxWidth": 1920, "maxHeight": 1080, "everyNthFrame": 1})
        await page.wait_for_timeout(1200)  # idle lead-in
        await page.evaluate("window.startRun()")
        result = None
        while result is None:
            await page.wait_for_timeout(250)
            result = await page.evaluate("window.__result || null")
        await page.wait_for_timeout(int(HOLD_END * 1000))
        await cdp.send("Page.stopScreencast")
        await page.wait_for_timeout(300)
        await page.screenshot(path=str(OUT.with_suffix(".png")))
        await browser.close()
    srv.shutdown()
    print("result", json.dumps(result), "frames", len(frames))
    # concat demuxer with real inter-frame durations -> constant 60 fps H.264
    lst = FRAMES / "list.txt"
    with lst.open("w") as f:
        for (ts, path), nxt in zip(frames, frames[1:] + [(frames[-1][0] + 1 / 60, None)]):
            f.write(f"file '{path.name}'\nduration {max(nxt[0] - ts, 1 / 120):.6f}\n")
        f.write(f"file '{frames[-1][1].name}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-vf", "fps=60,scale=in_range=pc:out_range=tv,format=yuv420p", "-color_range", "tv", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                    "-movflags", "+faststart", str(OUT)], check=True, cwd=FRAMES)
    span = frames[-1][0] - frames[0][0]
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {span:.1f} s captured)")
    (OUT.with_suffix(".json")).write_text(json.dumps(result, indent=1))


asyncio.run(main())
