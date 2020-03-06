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
    "no-underscore-dangle": ["error", {"allowAfterThis": true}],
    "no-console": ["error", {allow: ["warn", "error"]}],
    "no-param-reassign": ["error", {"props": false}],
    "class-methods-use-this": ["error", {"exceptMethods": ["init"]}],
    "no-unused-vars": ["error", {"varsIgnorePattern": "Api"}],
    "no-plusplus": ["error", {"allowForLoopAfterthoughts": true}],
  },
  "parser": "babel-eslint"
};
