use std::rc::Rc;

///词语的类型
///
#[allow(dead_code)]
#[derive(Debug)]
pub enum TokenType {
    TokenUnknown,
    TokenInputNumber(usize),
    TokenComma,
    TokenAdd,
    TokenMul,
    TokenLeftBracket,
    TokenRightBracket,
    TokenSin,
}
impl Clone for TokenType {
    fn clone(&self) -> Self {
        match self {
            // 根据需要手动克隆每个变体
            TokenType::TokenUnknown => TokenType::TokenUnknown,
            TokenType::TokenInputNumber(num) => TokenType::TokenInputNumber(num.clone()),
            TokenType::TokenComma => TokenType::TokenComma,
            TokenType::TokenAdd => TokenType::TokenAdd,
            TokenType::TokenMul => TokenType::TokenMul,
            TokenType::TokenLeftBracket => TokenType::TokenLeftBracket,
            TokenType::TokenRightBracket => TokenType::TokenRightBracket,
            TokenType::TokenSin => TokenType::TokenSin,
            // 其他变体的克隆逻辑
        }
    }
}
/// 词语Token
#[allow(dead_code)]
pub struct Token {
    token_type: TokenType,
    start_pos: usize,
    end_pose: usize,
}

/// 语法树的节点
#[allow(dead_code)]
pub struct TokenNode {
    pub num_index: TokenType,
    left: Option<Rc<TokenNode>>,
    right: Option<Rc<TokenNode>>,
}
#[allow(dead_code)]
pub struct ExpressionParser {
    pub tokens: Vec<Token>,
    token_strs: Vec<String>,
    statement: String,
}

impl ExpressionParser {
    fn remove_whitespace(statement: String) -> String {
        statement.chars().filter(|&c| !c.is_whitespace()).collect()
    }
    pub fn from_string(statement: &String) -> Self {
        let statement = statement.clone();

        //去掉空格
        let statement: String = Self::remove_whitespace(statement);
        //判断是否为空
        assert!(!statement.is_empty(), "The input statement is empty!");
        print!("{}", &statement);

        ExpressionParser {
            tokens: Vec::new(),
            token_strs: Vec::new(),
            statement: statement.clone(),
        }
    }
    pub fn token_strs(&self) -> &Vec<String> {
        &self.token_strs
    }
}
impl ExpressionParser {
    fn tokenizer_add(&mut self, chars_vec: &Vec<char>, s_idx: &mut usize) {
        assert!(*s_idx + 1 < chars_vec.len() && chars_vec[*s_idx + 1] == 'd');
        assert!(*s_idx + 2 < chars_vec.len() && chars_vec[*s_idx + 2] == 'd');
        let add_token = Token {
            token_type: TokenType::TokenAdd,
            start_pos: *s_idx,
            end_pose: *s_idx + 3,
        };

        self.tokens.push(add_token);
        *s_idx = *s_idx + 3;
        self.token_strs.push("add".to_string());
    }
    fn tokenizer_mul(&mut self, chars_vec: &Vec<char>, s_idx: &mut usize) {
        assert!(*s_idx + 1 < chars_vec.len() && chars_vec[*s_idx + 1] == 'u');
        assert!(*s_idx + 2 < chars_vec.len() && chars_vec[*s_idx + 2] == 'l');
        let mul_token = Token {
            token_type: TokenType::TokenMul,
            start_pos: *s_idx,
            end_pose: *s_idx + 3,
        };

        self.tokens.push(mul_token);
        *s_idx = *s_idx + 3;
        self.token_strs.push("mul".to_string());
    }
    fn tokenizer_number(&mut self, chars_vec: &Vec<char>, s_idx: &mut usize) {
        let mut cur_token_string = String::new();
        cur_token_string.push(chars_vec[*s_idx]);
        assert!(*s_idx + 1 < chars_vec.len() && chars_vec[*s_idx + 1].is_digit(10));

        let mut j = *s_idx + 1;
        while j < chars_vec.len() {
            if !chars_vec[j].is_digit(10) {
                break;
            }
            cur_token_string.push(chars_vec[j]);
            j = j + 1;
        }
        let input_num: usize = cur_token_string[1..].parse().unwrap();
        let number_token = Token {
            token_type: TokenType::TokenInputNumber(input_num),
            start_pos: *s_idx,
            end_pose: j,
        };
        self.tokens.push(number_token);
        self.token_strs.push(cur_token_string);
        *s_idx = j;
    }
    fn token_comma(&mut self, _chars_vec: &Vec<char>, s_idx: &mut usize) {
        let comma_token = Token {
            token_type: TokenType::TokenComma,
            start_pos: *s_idx,
            end_pose: *s_idx + 1,
        };
        self.tokens.push(comma_token);
        self.token_strs.push(",".to_string());
        *s_idx = *s_idx + 1;
    }
    fn token_left_bracket(&mut self, _chars_vec: &Vec<char>, s_idx: &mut usize) {
        let left_bracker_token = Token {
            token_type: TokenType::TokenLeftBracket,
            start_pos: *s_idx,
            end_pose: *s_idx + 1,
        };
        self.tokens.push(left_bracker_token);
        self.token_strs.push("(".to_string());
        *s_idx = *s_idx + 1;
    }
    fn token_right_bracket(&mut self, _chars_vec: &Vec<char>, s_idx: &mut usize) {
        let left_bracker_token = Token {
            token_type: TokenType::TokenRightBracket,
            start_pos: *s_idx,
            end_pose: *s_idx + 1,
        };
        self.tokens.push(left_bracker_token);
        self.token_strs.push(")".to_string());
        *s_idx = *s_idx + 1;
    }

    pub fn tokenizer(&mut self) {
        let statement_chars_vec: Vec<char> = self.statement.chars().collect();

        let mut statement_idx = 0;
        while statement_idx < statement_chars_vec.len() {
            match statement_chars_vec[statement_idx] {
                'a' => {
                    self.tokenizer_add(&statement_chars_vec, &mut statement_idx);
                }
                'm' => {
                    self.tokenizer_mul(&statement_chars_vec, &mut statement_idx);
                }
                '@' => {
                    self.tokenizer_number(&statement_chars_vec, &mut statement_idx);
                }
                ',' => {
                    self.token_comma(&statement_chars_vec, &mut statement_idx);
                }
                '(' => {
                    self.token_left_bracket(&statement_chars_vec, &mut statement_idx);
                }
                ')' => {
                    self.token_right_bracket(&statement_chars_vec, &mut statement_idx);
                }
                _ => {
                    panic!("Unknown  illegal character: ");
                }
            }
        }
    }
}
impl ExpressionParser {
    fn generate_number_token_node(&self, token: &Token) -> Rc<TokenNode> {
        let start_pose = token.start_pos + 1;
        let end_pose = token.end_pose;
        assert!(start_pose < end_pose);
        assert!(end_pose <= self.statement.len());

        let str_number = &self.statement[start_pose..end_pose];
        let num: usize = str_number.parse().expect("Failed to parse number");
        Rc::new(TokenNode {
            num_index: TokenType::TokenInputNumber(num),
            left: None,
            right: None,
        })
    }

    pub fn generate_(&self, index: &mut usize) -> Rc<TokenNode> {
        assert!(*index < self.tokens.len());
        let cur_token_type = self.tokens.get(*index).unwrap();
        matches!(
            cur_token_type.token_type,
            TokenType::TokenAdd | TokenType::TokenMul | TokenType::TokenInputNumber(_)
        );

        match cur_token_type.token_type {
            TokenType::TokenInputNumber(_) => self.generate_number_token_node(cur_token_type),
            TokenType::TokenAdd | TokenType::TokenMul => {
                let mut cur_node = TokenNode {
                    num_index: cur_token_type.token_type.clone(),
                    left: None,
                    right: None,
                };
                *index = *index + 1;
                assert!(*index < self.tokens.len());

                matches!(
                    self.tokens.get(*index).unwrap().token_type,
                    TokenType::TokenLeftBracket
                );
                *index = *index + 1;
                assert!(*index < self.tokens.len(), "Missing correspond left token!");
                let left_token = self.tokens.get(*index).unwrap();
                match left_token.token_type {
                    TokenType::TokenInputNumber(_) | TokenType::TokenAdd | TokenType::TokenMul => {
                        cur_node.left = Some(self.generate_(index));
                    }
                    _ => {
                        panic!("Unknown token type: {:?}", left_token.token_type);
                    }
                }
                *index = *index + 1;
                assert!(*index < self.tokens.len(), "Missing comma!");
                matches!(
                    self.tokens.get(*index).unwrap().token_type,
                    TokenType::TokenComma
                );

                *index = *index + 1;
                assert!(
                    *index < self.tokens.len(),
                    "Missing correspond right token!"
                );
                matches!(
                    self.tokens.get(*index).unwrap().token_type,
                    TokenType::TokenRightBracket
                );
                let right_token = self.tokens.get(*index).unwrap();
                match right_token.token_type {
                    TokenType::TokenInputNumber(_) | TokenType::TokenAdd | TokenType::TokenMul => {
                        cur_node.right = Some(self.generate_(index));
                    }
                    _ => {
                        panic!("Unknown token type: {:?}", right_token.token_type);
                    }
                }
                *index = *index + 1;
                assert!(*index < self.tokens.len());
                matches!(
                    self.tokens.get(*index).unwrap().token_type,
                    TokenType::TokenRightBracket
                );
                Rc::new(cur_node)
            }
            _ => {
                panic!("Unknown token type: {:?}", cur_token_type.token_type);
            }
        }
    }
    fn reverse_polish(root_node: &Option<Rc<TokenNode>>, reverse_polish: &mut Vec<Rc<TokenNode>>) {
        if let None = root_node {
            return;
        }

        Self::reverse_polish(&root_node.as_ref().unwrap().left, reverse_polish);
        Self::reverse_polish(&root_node.as_ref().unwrap().right, reverse_polish);
        reverse_polish.push(root_node.as_ref().unwrap().clone());
    }
    pub fn generate(&self) -> Vec<Rc<TokenNode>> {
        let mut index: usize = 0;
        let root = self.generate_(&mut index);

        //转逆波兰式,之后转移到expression中
        let mut reverse_polish = Vec::new();

        Self::reverse_polish(&Some(root), &mut reverse_polish);
        reverse_polish
    }
}
#[cfg(test)]
mod test_parse {
    use super::*;

    #[test]
    fn test_simple_parser() {
        let statement = "add(@0,mul(@1,@2))".to_string();
        let mut parser = ExpressionParser::from_string(&statement);
        parser.tokenizer();

        let tokens = &parser.tokens;

        let token_strs = parser.token_strs();

        assert_eq!(token_strs.get(0).unwrap(), &"add".to_string());
        matches!(tokens.get(0).unwrap().token_type, TokenType::TokenAdd);

        assert_eq!(token_strs.get(1).unwrap(), &"(".to_string());
        matches!(
            tokens.get(1).unwrap().token_type,
            TokenType::TokenLeftBracket
        );

        assert_eq!(token_strs.get(2).unwrap(), &"@0".to_string());
        matches!(
            tokens.get(2).unwrap().token_type,
            TokenType::TokenInputNumber(_)
        );

        assert_eq!(token_strs.get(3).unwrap(), &",".to_string());
        matches!(tokens.get(3).unwrap().token_type, TokenType::TokenComma);

        assert_eq!(token_strs.get(4).unwrap(), &"mul".to_string());
        matches!(tokens.get(4).unwrap().token_type, TokenType::TokenMul);

        assert_eq!(token_strs.get(5).unwrap(), &"(".to_string());
        matches!(
            tokens.get(5).unwrap().token_type,
            TokenType::TokenLeftBracket
        );

        assert_eq!(token_strs.get(6).unwrap(), &"@1".to_string());
        matches!(
            tokens.get(6).unwrap().token_type,
            TokenType::TokenInputNumber(_)
        );

        assert_eq!(token_strs.get(7).unwrap(), &",".to_string());
        matches!(tokens.get(7).unwrap().token_type, TokenType::TokenComma);

        assert_eq!(token_strs.get(8).unwrap(), &"@2".to_string());
        matches!(
            tokens.get(8).unwrap().token_type,
            TokenType::TokenInputNumber(_)
        );

        assert_eq!(token_strs.get(9).unwrap(), &")".to_string());
        matches!(
            tokens.get(9).unwrap().token_type,
            TokenType::TokenRightBracket
        );

        assert_eq!(token_strs.get(10).unwrap(), &")".to_string());
        matches!(
            tokens.get(10).unwrap().token_type,
            TokenType::TokenRightBracket
        );
    }

    #[test]
    fn test_parser_generate1() {
        let statement = "add(@0,@1)".to_string();
        let mut parser = ExpressionParser::from_string(&statement);
        parser.tokenizer();
        let mut index: usize = 0;
        // 抽象语法树:
        //
        //       add
        //       /  \
        //     mul   @2
        //    /   \
        //  @0    @1
        let node = parser.generate_(&mut index);
        matches!(node.num_index, TokenType::TokenRightBracket);
        matches!(
            node.left.as_ref().unwrap().num_index,
            TokenType::TokenInputNumber(0)
        );
        matches!(
            node.right.as_ref().unwrap().num_index,
            TokenType::TokenInputNumber(1)
        );
    }
    #[test]
    fn test_parser_generate2() {
        let statement = "add(mul(@0,@1),@2)".to_string();
        let mut parser = ExpressionParser::from_string(&statement);
        parser.tokenizer();
        let mut index: usize = 0;
        // 抽象语法树:
        //
        //       add
        //       /  \
        //     mul   @2
        //    /   \
        //  @0    @1
        let node = parser.generate_(&mut index);
        matches!(node.num_index, TokenType::TokenRightBracket);
        matches!(node.left.as_ref().unwrap().num_index, TokenType::TokenMul);
        matches!(
            node.left.as_ref().unwrap().left.as_ref().unwrap().num_index,
            TokenType::TokenInputNumber(0)
        );
        matches!(
            node.left
                .as_ref()
                .unwrap()
                .right
                .as_ref()
                .unwrap()
                .num_index,
            TokenType::TokenInputNumber(1)
        );

        matches!(
            node.right.as_ref().unwrap().num_index,
            TokenType::TokenInputNumber(2)
        );
    }

    #[test]
    fn test_parser_reverse_polish() {
        let statement = "add(mul(@0,@1),@2)".to_string();
        let mut parser = ExpressionParser::from_string(&statement);
        parser.tokenizer();

        // 抽象语法树:
        //
        //       add
        //       /  \
        //     mul   @2
        //    /   \
        //  @0    @1
        let reverse_polish = parser.generate();
        for item in &reverse_polish {
            println!("\n{:?}", item.as_ref().num_index);
        }
    }
}

#[cfg(test)]
mod test_rust {

    #[test]
    fn test_string_slice() {
        let s = String::from("#123456");
        let slice_s = &s[1..];
        println!("{}", slice_s);
    }
}
