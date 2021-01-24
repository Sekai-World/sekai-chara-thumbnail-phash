const sharp = require("sharp");

function initCOS(N = 64) {
  const entries = 2 * N * (N - 1);
  const COS = new Float64Array(entries);
  for (let i = 0; i < entries; i++) {
    COS[i] = Math.cos((i / (2 * N)) * Math.PI);
  }
  return COS;
}

const COS = initCOS(32);

function hash(data, N = 64) {
  const greyScale = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const index = 4 * (N * i + j);
      greyScale[N * i + j] =
        0.299 * data[index + 0] +
        0.587 * data[index + 1] +
        0.114 * data[index + 2];
    }
  }
  const dct = applyDCT2(greyScale, N);
  const output = [];
  for (let x = 1; x <= 8; x++) {
    for (let y = 1; y <= 8; y++) {
      output.push(dct[32 * x + y]);
    }
  }
  const median = output.slice().sort((a, b) => a - b)[
    Math.floor(output.length / 2)
  ];
  for (let i = 0; i < output.length; i++) {
    output[i] = output[i] > median ? 1 : 0;
  }
  return output;
}

function applyDCT2(f, N = 64) {
  const c = new Float64Array(N);
  for (let i = 1; i < N; i++) c[i] = 1;
  c[0] = 1 / Math.sqrt(2);
  const F = new Float64Array(N * N);
  for (let u = 0; u < N; u++) {
    for (let v = 0; v < N; v++) {
      let sum = 0;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          sum += COS[(2 * i + 1) * u] * COS[(2 * j + 1) * v] * f[N * i + j];
        }
      }
      sum *= (c[u] * c[v]) / 4;
      F[N * u + v] = sum;
    }
  }
  return F;
}

async function workload(input) {
  // const input = "res021_no008_after_training.png";

  const data = await sharp(input)
    .extract({ width: 52, height: 52, left: 23, top: 24 })
    .resize(32, 32)
    .raw()
    .toBuffer();
  // .toFile(`crop_${input}`);

  const hashed = hash(data, 32).join("");

  return { [basename(input)]: hashed };
}

const fs = require("fs");
const { join, basename } = require("path");

const dir = process.argv[2];
Promise.all(
  fs
    .readdirSync(dir)
    .filter((base) => base.endsWith(".png"))
    .map((base) => workload(join(dir, base)))
).then((results) => {
  fs.writeFileSync("chara_hash.json", JSON.stringify(results, null));
  console.log(join(__dirname, "chara_hash.json"))
});
