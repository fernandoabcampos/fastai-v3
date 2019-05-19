from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

# export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
export_file_url = 'https://drive.google.com/uc?export=download&id=1JABYGBTHMoDzv0QFqW8IjAlMSM0E7drM'
export_file_name = 'export.pkl'

classes = ['Label:  1 ( pattern ) -  Argyle', 'Label:  2 ( style ) -  Asymmetric', 'Label:  3 ( category ) -  Athletic Pants', 'Label:  4 ( category ) -  Athletic Sets', 'Label:  5 ( category ) -  Athletic Shirts', 'Label:  6 ( category ) -  Athletic Shorts', 'Label:  7 ( neckline ) -  Backless Dresses', 'Label:  8 ( category ) -  Baggy Jeans', 'Label:  9 ( style ) -  Bandage', 'Label:  10 ( style ) -  Bandeaus', 'Label:  11 ( category ) -  Batwing Tops', 'Label:  12 ( category ) -  Beach & Swim Wear', 'Label:  13 ( style ) -  Beaded', 'Label:  14 ( color ) -  Beige', 'Label:  15 ( category ) -  Bikinis', 'Label:  16 ( category ) -  Binders', 'Label:  17 ( color ) -  Black', 'Label:  18 ( category ) -  Blouses', 'Label:  19 ( color ) -  Blue', 'Label:  20 ( style ) -  Bodycon', 'Label:  21 ( category ) -  Bodysuits', 'Label:  22 ( category ) -  Boots', 'Label:  23 ( category ) -  Bra Straps', 'Label:  24 ( color ) -  Bronze', 'Label:  25 ( color ) -  Brown', 'Label:  26 ( category ) -  Bubble Coats', 'Label:  27 ( category ) -  Business Shoes', 'Label:  28 ( pattern ) -  Camouflage', 'Label:  29 ( material ) -  Canvas', 'Label:  30 ( category ) -  Capes & Capelets', 'Label:  31 ( category ) -  Capri Pants', 'Label:  32 ( category ) -  Cardigans', 'Label:  33 ( category ) -  Cargo Pants', 'Label:  34 ( category ) -  Cargo Shorts', 'Label:  35 ( material ) -  Cashmere', 'Label:  36 ( category ) -  Casual Dresses', 'Label:  37 ( category ) -  Casual Pants', 'Label:  38 ( category ) -  Casual Shirts', 'Label:  39 ( category ) -  Casual Shoes', 'Label:  40 ( category ) -  Casual Shorts', 'Label:  41 ( material ) -  Chambray', 'Label:  42 ( pattern ) -  Checkered', 'Label:  43 ( pattern ) -  Chevron', 'Label:  44 ( material ) -  Chiffon', 'Label:  45 ( color ) -  Clear', 'Label:  46 ( category ) -  Cleats', 'Label:  47 ( category ) -  Clubbing Dresses', 'Label:  48 ( category ) -  Cocktail Dresses', 'Label:  49 ( neckline ) -  Collared', 'Label:  50 ( material ) -  Corduroy', 'Label:  51 ( category ) -  Corsets', 'Label:  52 ( category ) -  Costumes & Cosplay', 'Label:  53 ( material ) -  Cotton', 'Label:  54 ( style ) -  Criss Cross', 'Label:  55 ( pattern ) -  Crochet', 'Label:  56 ( category ) -  Crop Tops', 'Label:  57 ( category ) -  Custom Made Clothing', 'Label:  58 ( category ) -  Dance Wear', 'Label:  59 ( material ) -  Denim', 'Label:  60 ( category ) -  Drawstring Pants', 'Label:  61 ( category ) -  Dress Shirts', 'Label:  62 ( category ) -  Dresses', 'Label:  63 ( style ) -  Embroidered', 'Label:  64 ( category ) -  Fashion Sets', 'Label:  65 ( material ) -  Faux Fur', 'Label:  66 ( gender ) -  Female', 'Label:  67 ( material ) -  Flannel', 'Label:  68 ( category ) -  Flats', 'Label:  69 ( material ) -  Fleece', 'Label:  70 ( pattern ) -  Floral', 'Label:  71 ( category ) -  Formal Dresses', 'Label:  72 ( pattern ) -  Fringe', 'Label:  73 ( style ) -  Furry', 'Label:  74 ( pattern ) -  Galaxy', 'Label:  75 ( pattern ) -  Geometric', 'Label:  76 ( material ) -  Gingham', 'Label:  77 ( color ) -  Gold', 'Label:  78 ( color ) -  Gray', 'Label:  79 ( color ) -  Green', 'Label:  80 ( category ) -  Halter Tops', 'Label:  81 ( category ) -  Harem Pants', 'Label:  82 ( pattern ) -  Hearts', 'Label:  83 ( category ) -  Heels', 'Label:  84 ( pattern ) -  Herringbone', 'Label:  85 ( style ) -  Hi-Lo', 'Label:  86 ( category ) -  Hiking Boots', 'Label:  87 ( style ) -  Hollow-Out', 'Label:  88 ( category ) -  Hoodies & Sweatshirts', 'Label:  89 ( category ) -  Hosiery, Stockings, Tights', 'Label:  90 ( pattern ) -  Houndstooth', 'Label:  91 ( category ) -  Jackets', 'Label:  92 ( category ) -  Jeans', 'Label:  93 ( category ) -  Jerseys', 'Label:  94 ( category ) -  Jilbaab', 'Label:  95 ( category ) -  Jumpsuits Overalls & Rompers', 'Label:  96 ( category ) -  Kimonos', 'Label:  97 ( material ) -  Knit', 'Label:  98 ( material ) -  Lace', 'Label:  99 ( material ) -  Leather', 'Label:  100 ( category ) -  Leggings', 'Label:  101 ( pattern ) -  Leopard And Cheetah', 'Label:  102 ( material ) -  Linen', 'Label:  103 ( category ) -  Lingerie Sleepwear & Underwear', 'Label:  104 ( category ) -  Loafers & Slip-on Shoes', 'Label:  105 ( sleeve ) -  Long Sleeved', 'Label:  106 ( gender ) -  Male', 'Label:  107 ( pattern ) -  Marbled', 'Label:  108 ( color ) -  Maroon', 'Label:  109 ( category ) -  Maternity', 'Label:  110 ( pattern ) -  Mesh', 'Label:  111 ( color ) -  Multi Color', 'Label:  112 ( material ) -  Neoprene', 'Label:  113 ( gender ) -  Neutral', 'Label:  114 ( category ) -  Nightgowns', 'Label:  115 ( material ) -  Nylon', 'Label:  116 ( neckline ) -  Off The Shoulder', 'Label:  117 ( color ) -  Orange', 'Label:  118 ( material ) -  Organza', 'Label:  119 ( category ) -  Padded Bras', 'Label:  120 ( pattern ) -  Paisley', 'Label:  121 ( category ) -  Pajamas', 'Label:  122 ( category ) -  Party Dresses', 'Label:  123 ( category ) -  Pasties', 'Label:  124 ( material ) -  Patent', 'Label:  125 ( color ) -  Peach', 'Label:  126 ( category ) -  Peacoats', 'Label:  127 ( category ) -  Pencil Skirts', 'Label:  128 ( style ) -  Peplum', 'Label:  129 ( category ) -  Petticoats', 'Label:  130 ( pattern ) -  Pin Stripes', 'Label:  131 ( color ) -  Pink', 'Label:  132 ( pattern ) -  Plaid', 'Label:  133 ( style ) -  Pleated', 'Label:  134 ( material ) -  Plush', 'Label:  135 ( pattern ) -  Polka Dot', 'Label:  136 ( category ) -  Polos', 'Label:  137 ( material ) -  Polyester', 'Label:  138 ( style ) -  Printed', 'Label:  139 ( category ) -  Prom Dresses', 'Label:  140 ( sleeve ) -  Puff Sleeves', 'Label:  141 ( category ) -  Pullover Sweaters', 'Label:  142 ( color ) -  Purple', 'Label:  143 ( pattern ) -  Quilted', 'Label:  144 ( neckline ) -  Racerback', 'Label:  145 ( category ) -  Rain Boots', 'Label:  146 ( category ) -  Raincoats', 'Label:  147 ( material ) -  Rayon', 'Label:  148 ( color ) -  Red', 'Label:  149 ( style ) -  Reversible', 'Label:  150 ( style ) -  Rhinestone Studded', 'Label:  151 ( pattern ) -  Ripped', 'Label:  152 ( category ) -  Robes', 'Label:  153 ( neckline ) -  Round Neck', 'Label:  154 ( pattern ) -  Ruched', 'Label:  155 ( pattern ) -  Ruffles', 'Label:  156 ( category ) -  Running Shoes', 'Label:  157 ( category ) -  Sandals', 'Label:  158 ( material ) -  Satin', 'Label:  159 ( pattern ) -  Sequins', 'Label:  160 ( category ) -  Sheer Tops', 'Label:  161 ( category ) -  Shoe Accessories', 'Label:  162 ( category ) -  Shoe Inserts', 'Label:  163 ( category ) -  Shoelaces', 'Label:  164 ( sleeve ) -  Short Sleeves', 'Label:  165 ( category ) -  Shorts', 'Label:  166 ( neckline ) -  Shoulder Drapes', 'Label:  167 ( material ) -  Silk', 'Label:  168 ( color ) -  Silver', 'Label:  169 ( category ) -  Skinny Jeans', 'Label:  170 ( category ) -  Skirts', 'Label:  171 ( sleeve ) -  Sleeveless', 'Label:  172 ( category ) -  Slippers', 'Label:  173 ( pattern ) -  Snakeskin', 'Label:  174 ( category ) -  Sneakers', 'Label:  175 ( style ) -  Spaghetti Straps', 'Label:  176 ( material ) -  Spandex', 'Label:  177 ( category ) -  Sports Bras', 'Label:  178 ( neckline ) -  Square Necked', 'Label:  179 ( category ) -  Stilettos', 'Label:  180 ( sleeve ) -  Strapless', 'Label:  181 ( pattern ) -  Stripes', 'Label:  182 ( material ) -  Suede', 'Label:  183 ( category ) -  Suits & Blazers', 'Label:  184 ( style ) -  Summer', 'Label:  185 ( category ) -  Sweatpants', 'Label:  186 ( neckline ) -  Sweetheart Neckline', 'Label:  187 ( category ) -  Swim Trunks', 'Label:  188 ( category ) -  Swimsuit Cover-ups', 'Label:  189 ( category ) -  Swimsuits', 'Label:  190 ( category ) -  T-Shirts', 'Label:  191 ( material ) -  Taffeta', 'Label:  192 ( color ) -  Tan', 'Label:  193 ( category ) -  Tank Tops', 'Label:  194 ( color ) -  Teal', 'Label:  195 ( category ) -  Thermal Underwear', 'Label:  196 ( category ) -  Thigh Highs', 'Label:  197 ( category ) -  Thongs', 'Label:  198 ( category ) -  Three Piece Suits', 'Label:  199 ( pattern ) -  Tie Dye', 'Label:  200 ( category ) -  Trench Coats', 'Label:  201 ( category ) -  Trousers', 'Label:  202 ( category ) -  Tube Tops', 'Label:  203 ( material ) -  Tulle', 'Label:  204 ( style ) -  Tunic', 'Label:  205 ( neckline ) -  Turtlenecks', 'Label:  206 ( category ) -  Tutus', 'Label:  207 ( material ) -  Tweed', 'Label:  208 ( material ) -  Twill', 'Label:  209 ( style ) -  Two-Tone', 'Label:  210 ( neckline ) -  U-Necks', 'Label:  211 ( category ) -  Undershirts', 'Label:  212 ( category ) -  Underwear', 'Label:  213 ( category ) -  Uniforms', 'Label:  214 ( neckline ) -  V-Necks', 'Label:  215 ( material ) -  Velour', 'Label:  216 ( material ) -  Velvet', 'Label:  217 ( category ) -  Vests', 'Label:  218 ( style ) -  Vintage Retro', 'Label:  219 ( material ) -  Vinyl', 'Label:  220 ( category ) -  Wedding Dresses', 'Label:  221 ( category ) -  Wedges & Platforms', 'Label:  222 ( color ) -  White', 'Label:  223 ( category ) -  Winter Boots', 'Label:  224 ( material ) -  Wool', 'Label:  225 ( style ) -  Wrap', 'Label:  226 ( color ) -  Yellow', 'Label:  227 ( category ) -  Yoga Pants', 'Label:  228 ( pattern ) -  Zebra']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]

    result_split = str(prediction).split(";")
    valor_final = ''
    for r in result_split:
        valor_final += '<br /> ' + classes[int(r)]
    return JSONResponse({'result': str(valor_final)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
