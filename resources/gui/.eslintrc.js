module.exports = {
  env: {
    browser: true,
    es6: true,
  },
  extends: [
    'airbnb-base',
  ],
  globals: {
    Atomics: 'readonly',
    SharedArrayBuffer: 'readonly',
  },
  parserOptions: {
    ecmaVersion: 2018,
  },
  rules: {
    "no-underscore-dangle": ["error", { "allowAfterThis": true }],
    "no-console": ["error", { allow: ["warn", "error"] }],
    "no-param-reassign": ["error", { "props": false }]
  },
  "parser": "babel-eslint"
};
